const express = require('express');
const { spawn, exec } = require('child_process');
const http = require('http');
const fs = require('fs');

const app = express();
const PORT = process.env.PORT || 5000;
const LB_HOST = 'load-balancer';
const LB_PORT = 8080;

app.use(express.json());

let runningProcess = null;
let currentScenario = null;
let currentAlgorithm = null;
let originalAlgorithm = null;
let currentRunId = null;
let failureTimer = null;
let backend2Stopped = false;
let isStopping = false;
let startInProgress = false;
let activeRunContext = null;
let failureStopPromise = null;
let lastCleanupError = null;
let failureInjectionError = null;

function dockerCommand(command) {
  return new Promise((resolve, reject) => {
    exec(command, (err, stdout, stderr) => {
      if (err) {
        console.error(stderr || err.message);
        return reject(err);
      }
      resolve(stdout.trim());
    });
  });
}

async function stopBackend2() {
  if (backend2Stopped) return;

  console.log('[experiment-runner] Stopping backend-2...');
  await dockerCommand('docker stop backend-2');
  backend2Stopped = true;
  console.log('[experiment-runner] backend-2 stopped');
}

async function restartBackend2() {
  if (!backend2Stopped) return;

  console.log('[experiment-runner] Restarting backend-2...');

  await dockerCommand('docker start backend-2');

  // Only mark recovered after Docker confirms success.
  backend2Stopped = false;
  console.log('[experiment-runner] backend-2 restarted');
}

function getAlgorithm() {
  return new Promise((resolve, reject) => {
    const req = http.request({
      hostname: LB_HOST,
      port: LB_PORT,
      path: '/admin/algorithm',
      method: 'GET'
    }, (res) => {
      let data = '';

      res.on('data', chunk => data += chunk);

      res.on('end', () => {
        if (res.statusCode < 200 || res.statusCode >= 300) {
          return reject(new Error(`Algorithm read failed: ${data}`));
        }

        try {
          const result = JSON.parse(data);
          if (!['rr', 'lc', 'rl'].includes(result.algorithm)) {
            return reject(new Error(
              `Invalid algorithm returned by load balancer: ${result.algorithm}`
            ));
          }
          resolve(result);
        } catch {
          reject(new Error(`Invalid algorithm response: ${data}`));
        }
      });
    });

    req.on('error', reject);
    req.end();
  });
}

function setAlgorithm(algorithm) {
  return new Promise((resolve, reject) => {
    const body = JSON.stringify({ algorithm });

    const req = http.request({
      hostname: LB_HOST,
      port: LB_PORT,
      path: '/admin/algorithm',
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Content-Length': Buffer.byteLength(body)
      }
    }, (res) => {
      let data = '';

      res.on('data', chunk => data += chunk);
      res.on('end', () => {
        if (res.statusCode < 200 || res.statusCode >= 300) {
          return reject(new Error(`Algorithm switch failed: ${data}`));
        }

        try {
          resolve(JSON.parse(data));
        } catch {
          reject(new Error(`Invalid algorithm response: ${data}`));
        }
      });
    });

    req.on('error', reject);
    req.write(body);
    req.end();
  });
}

function clearFailureTimer() {
  if (failureTimer) {
    clearTimeout(failureTimer);
    failureTimer = null;
  }
}

function finishRun(context, code, err) {
  // Prevent duplicate cleanup from error and close events.
  if (context.finishPromise) {
    return context.finishPromise;
  }

  context.finishPromise = (async () => {
    const cleanupErrors = [];

    clearFailureTimer();

    if (err) {
      console.error('[k6] Process error:', err.message);
    } else {
      console.log(`[k6] Process exited with code ${code}`);
    }

    // Wait for a backend stop operation already in progress.
    if (failureStopPromise) {
      await failureStopPromise;
    }

    // Always attempt backend recovery.
    if (backend2Stopped) {
      try {
        await restartBackend2();
      } catch (error) {
        cleanupErrors.push(`Backend recovery failed: ${error.message}`);
        console.error('[experiment-runner]', cleanupErrors.at(-1));
      }
    }

    // Restore the algorithm that was active before this experiment.
    if (context.originalAlgorithm) {
      try {
        await setAlgorithm(context.originalAlgorithm);
        console.log(
          '[experiment-runner] Restored original algorithm:',
          context.originalAlgorithm
        );
        originalAlgorithm = null;
      } catch (error) {
        cleanupErrors.push(`Algorithm restoration failed: ${error.message}`);
        console.error('[experiment-runner]', cleanupErrors.at(-1));
        originalAlgorithm = context.originalAlgorithm;
      }
    }

    if (activeRunContext === context) {
      runningProcess = null;
      currentScenario = null;
      currentAlgorithm = null;
      currentRunId = null;
      activeRunContext = null;
    }

    lastCleanupError = cleanupErrors.length
      ? cleanupErrors.join('; ')
      : null;

    isStopping = false;
  })();

  return context.finishPromise;
}

app.get('/health', (req, res) => {
  res.json({ status: 'ok' });
});

app.get('/status', (req, res) => {
  res.json({
    running: runningProcess !== null,
    starting: startInProgress,
    scenario: currentScenario,
    algorithm: currentAlgorithm,
    run_id: currentRunId,
    backend2_stopped: backend2Stopped,
    cleanup_pending: backend2Stopped || originalAlgorithm !== null,
    cleanup_error: lastCleanupError,
    failure_injection_error: failureInjectionError
  });
});

app.post('/start', async (req, res) => {
  if (runningProcess !== null || startInProgress || activeRunContext !== null) {
    return res.status(409).json({
      status: 'error',
      message: 'An experiment is already running or starting',
      run_id: currentRunId
    });
  }

  if (backend2Stopped || originalAlgorithm !== null) {
    return res.status(409).json({
      status: 'error',
      message: 'Previous experiment cleanup is incomplete. Call /stop to retry recovery.',
      cleanup_error: lastCleanupError
    });
  }

  const requestedScenario = req.body?.scenario || 'steady';
  const scenario = requestedScenario === 'normal'
    ? 'steady'
    : requestedScenario;

  const algorithm = String(req.body?.algorithm || '').toLowerCase();
  const trial = String(req.body?.trial || 'trial1');

  if (!['steady', 'spike', 'failure'].includes(scenario)) {
    return res.status(400).json({
      status: 'error',
      message: 'scenario must be steady, spike, or failure'
    });
  }

  if (!['rr', 'lc', 'rl'].includes(algorithm)) {
    return res.status(400).json({
      status: 'error',
      message: 'algorithm must be rr, lc, or rl'
    });
  }

  if (!/^[a-zA-Z0-9_-]{1,40}$/.test(trial)) {
    return res.status(400).json({
      status: 'error',
      message: 'trial may contain only letters, numbers, underscores, and hyphens'
    });
  }

  startInProgress = true;

  const context = {
    originalAlgorithm: null,
    finishPromise: null,
    process: null
  };

  activeRunContext = context;

  try {
    // Capture the existing algorithm BEFORE switching.
    const previous = await getAlgorithm();
    context.originalAlgorithm = previous.algorithm;
    originalAlgorithm = previous.algorithm;

    const selected = await setAlgorithm(algorithm);

    console.log(
      '[experiment-runner] Selected algorithm:',
      selected.algorithm
    );

    fs.mkdirSync('/results', { recursive: true });

    const timestamp = new Date()
      .toISOString()
      .replace(/[-:]/g, '')
      .replace(/\./g, '')
      .replace('Z', '');

    const runId = `${timestamp}_${scenario}_${algorithm}_${trial}`;

    const env = {
      ...process.env,
      SCENARIO: scenario,
      ALGORITHM: algorithm,
      RUN_ID: runId
    };

    currentScenario = scenario;
    currentAlgorithm = algorithm;
    currentRunId = runId;
    isStopping = false;
    lastCleanupError = null;
    failureInjectionError = null;

    const args = [
      'run',
      '--out', `json=/results/${runId}.json`,
      `--summary-export=/results/${runId}_summary.json`,
      '/scripts/loadtest.js'
    ];

    const proc = spawn('k6', args, { env });

    context.process = proc;
    runningProcess = proc;

    proc.stdout.on('data', data => {
      process.stdout.write(`[k6] ${data}`);
    });

    proc.stderr.on('data', data => {
      process.stderr.write(`[k6 err] ${data}`);
    });

    proc.once('close', code => {
      finishRun(context, code);
    });

    proc.once('error', err => {
      finishRun(context, null, err);
    });

    if (scenario === 'failure') {
      failureTimer = setTimeout(() => {
        failureStopPromise = stopBackend2()
          .catch(error => {
            failureInjectionError = error.message;

            console.error(
              '[experiment-runner] Could not stop backend-2:',
              error.message
            );
          })
          .finally(() => {
            failureStopPromise = null;
          });
      }, 30000);
    }

    res.json({
      status: 'started',
      scenario,
      algorithm,
      trial,
      run_id: runId,
      raw_results: `/results/${runId}.json`,
      summary_results: `/results/${runId}_summary.json`
    });
  } catch (error) {
    console.error('[experiment-runner] Start failed:', error.message);

    await finishRun(context, null, error);

    res.status(500).json({
      status: 'error',
      message: error.message,
      cleanup_error: lastCleanupError
    });
  } finally {
    startInProgress = false;
  }
});

app.post('/stop', async (req, res) => {
  if (startInProgress && runningProcess === null) {
    return res.status(409).json({
      status: 'error',
      message: 'Experiment startup is still in progress. Retry /stop once startup completes.'
    });
  }

  clearFailureTimer();

  const context = activeRunContext;
  const proc = runningProcess;

  try {
    if (proc) {
      isStopping = true;

      // Wait for process termination and its close event.
      await new Promise(resolve => {
        let settled = false;

        const done = () => {
          if (settled) return;
          settled = true;
          clearTimeout(killTimer);
          resolve();
        };

        const killTimer = setTimeout(() => {
          if (!settled) {
            console.log('[experiment-runner] Sending SIGKILL to k6');
            proc.kill('SIGKILL');
          }
        }, 3000);

        proc.once('close', done);

        // Handle a process that has already exited.
        if (proc.exitCode !== null || proc.signalCode !== null) {
          done();
          return;
        }

        proc.kill('SIGINT');
      });
    }

    // Wait for normal close-event cleanup, or initiate cleanup
    // if the process had already exited.
    if (context) {
      await finishRun(context, null);
    }

    // Retry any incomplete recovery from a previous cleanup.
    if (failureStopPromise) {
      await failureStopPromise;
    }

    if (backend2Stopped) {
      await restartBackend2();
    }

    if (originalAlgorithm) {
      await setAlgorithm(originalAlgorithm);
      console.log(
        '[experiment-runner] Restored original algorithm:',
        originalAlgorithm
      );
      originalAlgorithm = null;
    }

    lastCleanupError = null;
    isStopping = false;

    return res.json({
      status: 'stopped',
      backend2_stopped: backend2Stopped,
      algorithm_restored: originalAlgorithm === null
    });
  } catch (error) {
    lastCleanupError = error.message;

    console.error(
      '[experiment-runner] Stop/recovery failed:',
      error.message
    );

    return res.status(500).json({
      status: 'error',
      message: error.message,
      backend2_stopped: backend2Stopped,
      original_algorithm_pending: originalAlgorithm
    });
  }
});

app.listen(PORT, '0.0.0.0', () => {
  console.log(`[experiment-runner] Server listening on port ${PORT}`);
});
