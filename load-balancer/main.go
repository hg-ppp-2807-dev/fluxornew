package main

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"net/http/httputil"
	"net/url"
	"os"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"github.com/gorilla/websocket"
	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promhttp"
)

// ─── Types ────────────────────────────────────────────────────────────────────

type Server struct {
	URL          *url.URL
	ID           int
	ActiveConns  int64
	Healthy      int32 // 1=healthy, 0=down
	TotalReqs    int64
	mu           sync.RWMutex
	latencySum   float64
	latencyCount int64
}

func (s *Server) IsHealthy() bool { return atomic.LoadInt32(&s.Healthy) == 1 }
func (s *Server) IncrConn()       { atomic.AddInt64(&s.ActiveConns, 1) }
func (s *Server) DecrConn()       { atomic.AddInt64(&s.ActiveConns, -1) }
func (s *Server) GetConns() int64 { return atomic.LoadInt64(&s.ActiveConns) }

func (s *Server) RecordLatency(ms float64) {
	s.mu.Lock()
	s.latencySum += ms
	s.latencyCount++
	s.mu.Unlock()
}

func (s *Server) AvgLatency() float64 {
	s.mu.RLock()
	defer s.mu.RUnlock()
	if s.latencyCount == 0 {
		return 0
	}
	return s.latencySum / float64(s.latencyCount)
}

// ─── Load Balancer ────────────────────────────────────────────────────────────

type LoadBalancer struct {
	servers    []*Server
	rrCounter  uint64
	algorithm  string
	algMu      sync.RWMutex
	rlAgentURL string
	hub        *Hub

	rateMu       sync.Mutex
	requestTimes []time.Time
}

func (lb *LoadBalancer) recordRequestAndGetRate() float64 {
	lb.rateMu.Lock()
	defer lb.rateMu.Unlock()

	now := time.Now()
	cutoff := now.Add(-time.Second)

	// Remove timestamps outside the rolling one-second window.
	firstValid := 0
	for firstValid < len(lb.requestTimes) &&
		lb.requestTimes[firstValid].Before(cutoff) {
		firstValid++
	}
	lb.requestTimes = lb.requestTimes[firstValid:]

	// Count this incoming request.
	lb.requestTimes = append(lb.requestTimes, now)

	// Requests observed in the rolling one-second window.
	return float64(len(lb.requestTimes))
}

func NewLoadBalancer(backendURLs []string, rlAgentURL string, hub *Hub) *LoadBalancer {
	servers := make([]*Server, len(backendURLs))
	for i, rawURL := range backendURLs {
		u, err := url.Parse(rawURL)
		if err != nil {
			log.Fatalf("invalid backend URL %s: %v", rawURL, err)
		}
		servers[i] = &Server{URL: u, ID: i, Healthy: 1}
	}
	return &LoadBalancer{
		servers:    servers,
		algorithm:  os.Getenv("ALGORITHM"),
		rlAgentURL: rlAgentURL,
		hub:        hub,
	}
}

func (lb *LoadBalancer) SetAlgorithm(algo string) {
	lb.algMu.Lock()
	lb.algorithm = algo
	lb.algMu.Unlock()
	log.Printf("[LB] Algorithm switched to: %s", algo)
}

func (lb *LoadBalancer) GetAlgorithm() string {
	lb.algMu.RLock()
	defer lb.algMu.RUnlock()
	return lb.algorithm
}

func (lb *LoadBalancer) healthyServers() []*Server {
	var healthy []*Server
	for _, s := range lb.servers {
		if s.IsHealthy() {
			healthy = append(healthy, s)
		}
	}
	return healthy
}

// Round Robin
func (lb *LoadBalancer) roundRobin() *Server {
	healthy := lb.healthyServers()
	if len(healthy) == 0 {
		return nil
	}
	idx := atomic.AddUint64(&lb.rrCounter, 1)
	return healthy[int(idx-1)%len(healthy)]
}

// Least Connections
func (lb *LoadBalancer) leastConnections() *Server {
	healthy := lb.healthyServers()
	if len(healthy) == 0 {
		return nil
	}
	best := healthy[0]
	for _, s := range healthy[1:] {
		if s.GetConns() < best.GetConns() {
			best = s
		}
	}
	return best
}

// RL Agent call
type RLState struct {
	CPUUtil     []float64 `json:"cpu_util"`
	ActiveConns []float64 `json:"active_conns"`
	AvgLatency  []float64 `json:"avg_latency_ms"`
	RequestRate float64   `json:"request_rate"`
	QueueDepth  float64   `json:"queue_depth"`
	HealthyMask []bool    `json:"healthy_mask"`
}

type RLResponse struct {
	Action int     `json:"action"`
	QValue float64 `json:"q_value"`
}

type rlFeedbackJob struct {
	RequestID   string
	ServerID    int
	LatencyMs   float64
	RequestRate float64
	QueueDepth  float64
	ActiveConns []float64
	AvgLatency  []float64
}

var rlFeedbackQueue chan rlFeedbackJob

const (
	rlFeedbackWorkers   = 8
	rlFeedbackQueueSize = 2048
)

var rlHTTPClient = &http.Client{
	Transport: &http.Transport{
		MaxIdleConns:        1000,
		MaxIdleConnsPerHost: 200,
		IdleConnTimeout:     90 * time.Second,
		DisableKeepAlives:   false,
	},
	Timeout: 50 * time.Millisecond,
}

var rlFeedbackHTTPClient = &http.Client{
	Transport: rlHTTPClient.Transport,
	Timeout:   500 * time.Millisecond,
}

func (lb *LoadBalancer) rlDecide(
	requestID string,
	requestRate float64,
	queueDepth float64,
) (*Server, string) {

	fallback := func(reason string) (*Server, string) {
		rlFallbacks.Inc()
		rlFallbackReasons.WithLabelValues(reason).Inc()
		return lb.roundRobin(), "rr_fallback"
	}

	healthy := lb.healthyServers()
	if len(healthy) == 0 {
		return fallback("no_healthy_backend")
	}

	healthyMask := make([]bool, len(lb.servers))

	state := RLState{
		CPUUtil:     make([]float64, len(lb.servers)),
		ActiveConns: make([]float64, len(lb.servers)),
		AvgLatency:  make([]float64, len(lb.servers)),
		RequestRate: requestRate,
		QueueDepth:  queueDepth,
		HealthyMask: healthyMask,
	}

	for i, s := range lb.servers {
		healthyMask[i] = s.IsHealthy()
		state.ActiveConns[i] = float64(s.GetConns())
		state.AvgLatency[i] = s.AvgLatency()
	}

	payload := map[string]interface{}{
		"request_id":     requestID,
		"cpu_util":       state.CPUUtil,
		"active_conns":   state.ActiveConns,
		"avg_latency_ms": state.AvgLatency,
		"request_rate":   state.RequestRate,
		"queue_depth":    state.QueueDepth,
		"healthy_mask":   healthyMask,
	}

	body, err := json.Marshal(payload)
	if err != nil {
		return fallback("marshal_error")
	}

	ctx, cancel := context.WithTimeout(context.Background(), 50*time.Millisecond)
	defer cancel()

	req, err := http.NewRequestWithContext(
		ctx,
		http.MethodPost,
		lb.rlAgentURL+"/decide",
		bytes.NewReader(body),
	)
	if err != nil {
		return fallback("request_error")
	}

	req.Header.Set("Content-Type", "application/json")

	decisionStart := time.Now()
	resp, err := rlHTTPClient.Do(req)
	rlDecisionLatency.Observe(float64(time.Since(decisionStart).Microseconds()) / 1000.0)

	if err != nil {
		rlTimeouts.Inc()
		return fallback("agent_unavailable")
	}
	defer resp.Body.Close()

	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		return fallback("agent_http_error")
	}

	var rlResp RLResponse
	if err := json.NewDecoder(resp.Body).Decode(&rlResp); err != nil {
		return fallback("invalid_response")
	}

	action := rlResp.Action

	if action < 0 || action >= len(lb.servers) {
		return fallback("invalid_action")
	}

	if !lb.servers[action].IsHealthy() {
		return fallback("unhealthy_action")
	}

	return lb.servers[action], "rl"
}

func (lb *LoadBalancer) SelectServer(reqID string, requestRate float64, queueDepth float64) (*Server, string) {
	algo := lb.GetAlgorithm()

	switch algo {
	case "lc":
		return lb.leastConnections(), algo
	case "rl":
		return lb.rlDecide(reqID, requestRate, queueDepth)
	default:
		return lb.roundRobin(), "rr"
	}
}

// ─── WebSocket Hub ────────────────────────────────────────────────────────────

type Client struct {
	conn *websocket.Conn
	send chan []byte
}

type Hub struct {
	clients    map[*Client]bool
	broadcast  chan []byte
	register   chan *Client
	unregister chan *Client
	mu         sync.Mutex
}

func newHub() *Hub {
	return &Hub{
		clients:    make(map[*Client]bool),
		broadcast:  make(chan []byte, 256),
		register:   make(chan *Client),
		unregister: make(chan *Client),
	}
}

func (h *Hub) Run() {
	for {
		select {
		case c := <-h.register:
			h.mu.Lock()
			h.clients[c] = true
			h.mu.Unlock()
		case c := <-h.unregister:
			h.mu.Lock()
			if _, ok := h.clients[c]; ok {
				delete(h.clients, c)
				close(c.send)
			}
			h.mu.Unlock()
		case msg := <-h.broadcast:
			h.mu.Lock()
			for c := range h.clients {
				select {
				case c.send <- msg:
				default:
					delete(h.clients, c)
					close(c.send)
				}
			}
			h.mu.Unlock()
		}
	}
}

var upgrader = websocket.Upgrader{
	CheckOrigin: func(r *http.Request) bool { return true },
}

func (h *Hub) ServeWS(w http.ResponseWriter, r *http.Request) {
	conn, err := upgrader.Upgrade(w, r, nil)
	if err != nil {
		log.Printf("[WS] upgrade error: %v", err)
		return
	}
	client := &Client{conn: conn, send: make(chan []byte, 64)}
	h.register <- client

	// Writer goroutine
	go func() {
		defer func() {
			h.unregister <- client
			conn.Close()
		}()
		for msg := range client.send {
			if err := conn.WriteMessage(websocket.TextMessage, msg); err != nil {
				break
			}
		}
	}()

	// Reader goroutine (handle incoming commands from dashboard)
	go func() {
		defer func() { h.unregister <- client }()
		for {
			_, msg, err := conn.ReadMessage()
			if err != nil {
				break
			}
			// Commands from dashboard: {"type":"SET_ALGO","algo":"rl"}
			var cmd map[string]string
			if json.Unmarshal(msg, &cmd) == nil {
				if cmd["type"] == "SET_ALGO" {
					lb.SetAlgorithm(cmd["algo"])
				}
			}
		}
	}()
}

// ─── Prometheus Metrics ───────────────────────────────────────────────────────

var (
	reqTotal = prometheus.NewCounterVec(prometheus.CounterOpts{
		Name: "lb_requests_total",
		Help: "Total requests routed",
	}, []string{"server_id", "algorithm"})

	reqLatency = prometheus.NewHistogramVec(prometheus.HistogramOpts{
		Name:    "lb_request_latency_ms",
		Help:    "Request latency in ms",
		Buckets: []float64{5, 10, 25, 50, 100, 250, 500, 1000, 2000},
	}, []string{"server_id", "algorithm"})

	activeConns = prometheus.NewGaugeVec(prometheus.GaugeOpts{
		Name: "lb_active_connections",
		Help: "Active connections per backend",
	}, []string{"server_id"})

	rlTimeouts = prometheus.NewCounter(prometheus.CounterOpts{
		Name: "lb_rl_timeouts_total",
		Help: "Total timeouts calling RL agent",
	})

	rlFallbacks = prometheus.NewCounter(prometheus.CounterOpts{
		Name: "lb_rl_fallbacks_total",
		Help: "Total fallbacks from RL to RR",
	})

	rlDecisionLatency = prometheus.NewHistogram(prometheus.HistogramOpts{
		Name:    "lb_rl_decision_latency_ms",
		Help:    "Time taken for RL agent decisions in milliseconds",
		Buckets: []float64{1, 2, 5, 10, 20, 35, 50, 75, 100, 250},
	})

	rlFallbackReasons = prometheus.NewCounterVec(prometheus.CounterOpts{
		Name: "lb_rl_fallback_reasons_total",
		Help: "RL fallback count grouped by reason",
	}, []string{"reason"})

	rlFeedbackDropped = prometheus.NewCounter(prometheus.CounterOpts{
		Name: "lb_rl_feedback_dropped_total",
		Help: "RL feedback jobs dropped because the bounded queue was full",
	})

	rlFeedbackErrors = prometheus.NewCounterVec(
		prometheus.CounterOpts{
			Name: "lb_rl_feedback_errors_total",
			Help: "Failed RL feedback requests grouped by reason",
		},
		[]string{"reason"},
	)
)

func initMetrics() {
	prometheus.MustRegister(
		reqTotal,
		reqLatency,
		activeConns,
		rlTimeouts,
		rlFallbacks,
		rlDecisionLatency,
		rlFallbackReasons,
		rlFeedbackDropped,
		rlFeedbackErrors,
	)

	// Pre-initialize known fallback reasons so zero-valued series appear from startup.
	for _, reason := range []string{
		"no_healthy_backend",
		"marshal_error",
		"request_error",
		"agent_unavailable",
		"agent_http_error",
		"invalid_response",
		"invalid_action",
		"unhealthy_action",
	} {
		rlFallbackReasons.WithLabelValues(reason).Add(0)
	}

	// Pre-initialize known feedback error reasons so zero-valued series appear from startup.
	for _, reason := range []string{
		"request_error",
		"http_error",
		"response_read_error",
		"invalid_response",
		"transition_skipped",
	} {
		rlFeedbackErrors.WithLabelValues(reason).Add(0)
	}
}

// ─── Route Event ──────────────────────────────────────────────────────────────

type ServerState struct {
	ID         int     `json:"id"`
	ActiveConn int64   `json:"conns"`
	AvgLatency float64 `json:"latency_ms"`
	Healthy    bool    `json:"healthy"`
}

type RouteEvent struct {
	Type         string        `json:"type"`
	RequestID    string        `json:"request_id"`
	Timestamp    int64         `json:"timestamp"`
	Algorithm    string        `json:"algorithm"`
	ServerID     int           `json:"server_id"`
	LatencyMs    float64       `json:"latency_ms"`
	ServerStates []ServerState `json:"server_states"`
}

// Global lb reference needed for WS command handler
var lb *LoadBalancer
var inFlightRequests int64
var reqSeq uint64

// ─── Main Handler ─────────────────────────────────────────────────────────────

func proxyHandler(w http.ResponseWriter, r *http.Request) {
	atomic.AddInt64(&inFlightRequests, 1)
	defer atomic.AddInt64(&inFlightRequests, -1)

	start := time.Now()
	reqID := fmt.Sprintf("req_%d_%d", start.UnixMilli(), atomic.AddUint64(&reqSeq, 1))
	requestRate := lb.recordRequestAndGetRate()
	queueDepth := atomic.LoadInt64(&inFlightRequests) - 1

	server, algo := lb.SelectServer(reqID, requestRate, float64(queueDepth))
	if server == nil {
		http.Error(w, "no healthy backends", http.StatusServiceUnavailable)
		return
	}

	server.IncrConn()
	activeConns.WithLabelValues(fmt.Sprintf("%d", server.ID)).Set(float64(server.GetConns()))

	proxy := httputil.NewSingleHostReverseProxy(server.URL)
	proxy.ErrorHandler = func(w http.ResponseWriter, r *http.Request, err error) {
		log.Printf("[PROXY] error to %s: %v", server.URL, err)
		atomic.StoreInt32(&server.Healthy, 0)
		http.Error(w, "backend error", http.StatusBadGateway)
	}

	proxy.ServeHTTP(w, r)

	latency := float64(time.Since(start).Milliseconds())
	server.DecrConn()
	server.RecordLatency(latency)
	atomic.AddInt64(&server.TotalReqs, 1)

	activeConns.WithLabelValues(fmt.Sprintf("%d", server.ID)).Set(float64(server.GetConns()))
	reqTotal.WithLabelValues(fmt.Sprintf("%d", server.ID), algo).Inc()
	reqLatency.WithLabelValues(fmt.Sprintf("%d", server.ID), algo).Observe(latency)

	// Send feedback to RL agent (non-blocking)
	if algo == "rl" {
		enqueueRLFeedback(
			reqID,
			server.ID,
			latency,
			requestRate,
			float64(queueDepth),
			lb.servers,
		)
	}

	// Broadcast WebSocket event
	states := make([]ServerState, len(lb.servers))
	for i, s := range lb.servers {
		states[i] = ServerState{
			ID:         s.ID,
			ActiveConn: s.GetConns(),
			AvgLatency: s.AvgLatency(),
			Healthy:    s.IsHealthy(),
		}
	}
	event := RouteEvent{
		Type:         "ROUTE_EVENT",
		RequestID:    reqID,
		Timestamp:    start.UnixMilli(),
		Algorithm:    algo,
		ServerID:     server.ID,
		LatencyMs:    latency,
		ServerStates: states,
	}
	if data, err := json.Marshal(event); err == nil {
		lb.hub.broadcast <- data
	}
}

func startRLFeedbackWorkers() {
	rlFeedbackQueue = make(chan rlFeedbackJob, rlFeedbackQueueSize)

	for i := 0; i < rlFeedbackWorkers; i++ {
		go func() {
			for job := range rlFeedbackQueue {
				sendRLFeedback(job)
			}
		}()
	}

	log.Printf(
		"[LB] Started %d RL feedback workers (queue capacity: %d)",
		rlFeedbackWorkers,
		rlFeedbackQueueSize,
	)
}

func enqueueRLFeedback(
	requestID string,
	serverID int,
	latencyMs float64,
	requestRate float64,
	queueDepth float64,
	servers []*Server,
) {
	job := rlFeedbackJob{
		RequestID:   requestID,
		ServerID:    serverID,
		LatencyMs:   latencyMs,
		RequestRate: requestRate,
		QueueDepth:  queueDepth,
		ActiveConns: make([]float64, len(servers)),
		AvgLatency:  make([]float64, len(servers)),
	}

	// Capture the next-state snapshot before queueing.
	for i, s := range servers {
		job.ActiveConns[i] = float64(s.GetConns())
		job.AvgLatency[i] = s.AvgLatency()
	}

	select {
	case rlFeedbackQueue <- job:
	default:
		rlFeedbackDropped.Inc()
	}
}

func sendRLFeedback(job rlFeedbackJob) {
	body, err := json.Marshal(map[string]interface{}{
		"request_id":     job.RequestID,
		"server_id":      job.ServerID,
		"latency_ms":     job.LatencyMs,
		"active_conns":   job.ActiveConns,
		"avg_latency_ms": job.AvgLatency,
		"request_rate":   job.RequestRate,
		"queue_depth":    job.QueueDepth,
	})
	if err != nil {
		return
	}

	ctx, cancel := context.WithTimeout(context.Background(), 100*time.Millisecond)
	defer cancel()

	req, err := http.NewRequestWithContext(
		ctx,
		http.MethodPost,
		lb.rlAgentURL+"/feedback",
		bytes.NewReader(body),
	)
	if err != nil {
		return
	}

	req.Header.Set("Content-Type", "application/json")

	resp, err := rlFeedbackHTTPClient.Do(req)
	if err != nil {
		rlFeedbackErrors.WithLabelValues("request_error").Inc()
		return
	}
	defer resp.Body.Close()

	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		rlFeedbackErrors.WithLabelValues("http_error").Inc()
		return
	}

	responseBody, err := io.ReadAll(io.LimitReader(resp.Body, 4096))
	if err != nil {
		rlFeedbackErrors.WithLabelValues("response_read_error").Inc()
		return
	}

	var feedbackResponse struct {
		Status string `json:"status"`
		Reason string `json:"reason"`
	}

	if err := json.Unmarshal(responseBody, &feedbackResponse); err != nil {
		rlFeedbackErrors.WithLabelValues("invalid_response").Inc()
		return
	}

	if feedbackResponse.Status != "ok" {
		rlFeedbackErrors.WithLabelValues("transition_skipped").Inc()
	}
}

// ─── Health Checker ───────────────────────────────────────────────────────────

func healthChecker(servers []*Server) {
	ticker := time.NewTicker(1 * time.Second)
	for range ticker.C {
		for _, s := range servers {
			go func(srv *Server) {
				ctx, cancel := context.WithTimeout(context.Background(), 1*time.Second)
				defer cancel()
				req, _ := http.NewRequestWithContext(ctx, "GET", srv.URL.String()+"/health", nil)
				resp, err := http.DefaultClient.Do(req)
				if err != nil || resp.StatusCode != 200 {
					if atomic.SwapInt32(&srv.Healthy, 0) == 1 {
						log.Printf("[HEALTH] Server %d UNHEALTHY", srv.ID)
					}
				} else {
					if atomic.SwapInt32(&srv.Healthy, 1) == 0 {
						log.Printf("[HEALTH] Server %d RECOVERED", srv.ID)
					}
				}
				if resp != nil {
					resp.Body.Close()
				}
			}(s)
		}
	}
}

// ─── Main ─────────────────────────────────────────────────────────────────────

func main() {
	initMetrics()

	backendURLs := strings.Split(os.Getenv("BACKEND_URLS"), ",")
	rlAgentURL := os.Getenv("RL_AGENT_URL")
	if rlAgentURL == "" {
		rlAgentURL = "http://localhost:5000"
	}

	hub := newHub()
	go hub.Run()

	lb = NewLoadBalancer(backendURLs, rlAgentURL, hub)

	startRLFeedbackWorkers()
	go healthChecker(lb.servers)

	mux := http.NewServeMux()

	// WebSocket endpoint
	mux.HandleFunc("/ws", hub.ServeWS)

	// Prometheus metrics
	mux.Handle("/metrics", promhttp.Handler())

	// Admin API
	mux.HandleFunc("/admin/algorithm", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Access-Control-Allow-Origin", "http://localhost:3000")
		w.Header().Set("Access-Control-Allow-Methods", "POST, GET, OPTIONS")
		w.Header().Set("Access-Control-Allow-Headers", "Content-Type")

		if r.Method == http.MethodOptions {
			w.WriteHeader(http.StatusNoContent)
			return
		}

		if r.Method == http.MethodPost {
			var body struct {
				Algorithm string `json:"algorithm"`
			}
			json.NewDecoder(r.Body).Decode(&body)
			lb.SetAlgorithm(body.Algorithm)
			w.Header().Set("Content-Type", "application/json")
			json.NewEncoder(w).Encode(map[string]string{"algorithm": body.Algorithm})
			return
		}
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]string{"algorithm": lb.GetAlgorithm()})
	})

	mux.HandleFunc("/admin/status", func(w http.ResponseWriter, r *http.Request) {
		type Status struct {
			Algorithm string        `json:"algorithm"`
			Servers   []ServerState `json:"servers"`
		}
		states := make([]ServerState, len(lb.servers))
		for i, s := range lb.servers {
			states[i] = ServerState{
				ID:         s.ID,
				ActiveConn: s.GetConns(),
				AvgLatency: s.AvgLatency(),
				Healthy:    s.IsHealthy(),
			}
		}
		w.Header().Set("Content-Type", "application/json")
		w.Header().Set("Access-Control-Allow-Origin", "*")
		json.NewEncoder(w).Encode(Status{Algorithm: lb.GetAlgorithm(), Servers: states})
	})

	// Proxy all other requests
	mux.HandleFunc("/", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Access-Control-Allow-Origin", "*")
		if r.Method == http.MethodOptions {
			w.WriteHeader(http.StatusNoContent)
			return
		}
		proxyHandler(w, r)
	})

	log.Println("[LB] Load balancer starting on :8080")
	if err := http.ListenAndServe(":8080", mux); err != nil {
		log.Fatal(err)
	}
}
