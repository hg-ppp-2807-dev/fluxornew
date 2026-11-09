import http from 'k6/http';
import { sleep, check } from 'k6';

const LB = __ENV.LB_URL || 'http://localhost:8080';
const scenario = __ENV.SCENARIO || 'steady';

// Steady traffic: 50 VUs for 60 seconds
const steadyOptions = {
  scenarios: {
    steady: {
      executor: 'constant-vus',
      vus: 50,
      duration: '60s',
    },
  },
  thresholds: {
    http_req_duration: ['p(95)<500'],
    http_req_failed: ['rate<0.01'],
  },
};

// Spike: ramp 10 to 200, hold, then ramp down
const spikeOptions = {
  scenarios: {
    spike: {
      executor: 'ramping-vus',
      startVUs: 10,
      stages: [
        { duration: '30s', target: 200 },
        { duration: '30s', target: 200 },
        { duration: '20s', target: 10 },
      ],
    },
  },
};

// Partial failure: 50 VUs for 90 seconds
const failureOptions = {
  scenarios: {
    failure_test: {
      executor: 'constant-vus',
      vus: 50,
      duration: '90s',
    },
  },
  thresholds: {
    http_req_failed: ['rate<0.05'],
  },
};

export const options =
  scenario === 'steady' || scenario === 'normal'
    ? steadyOptions
    : scenario === 'failure'
      ? failureOptions
      : spikeOptions;

export default function () {
  const res = http.get(`${LB}/work`, { timeout: '5s' });

  check(res, {
    'status is 200': r => r.status === 200,
    'latency < 1s': r => r.timings.duration < 1000,
  });

  sleep(0.1);
}
