/* ═══ Sticky Bottom Event Log Bar — Shared JS (t_e904d515) ═══ */
/* Auto-initializes on any page that includes this script + the HTML container */
/* Usage:
     <link rel="stylesheet" href="/admin/static/event_log_bar.css">
     <div id="elb-container" data-draft-id="{{ draft.id }}"></div>
     <script src="/admin/static/event_log_bar.js"></script>
   Set data-draft-id to filter events for a specific draft; omit for all events.
*/

(function() {
    'use strict';

    const POLL_INTERVAL = 3000; // 3 seconds
    const MAX_ENTRIES = 100;

    function initEventLogBar() {
        const container = document.getElementById('elb-container');
        if (!container) return;

        const draftId = container.getAttribute('data-draft-id') || null;

        // Build the HTML structure
        container.innerHTML = `
            <div class="elb-wrapper" id="elb-wrapper">
                <div class="elb-resize-handle" id="elb-resize-handle" title="Zum Anpassen der Höhe ziehen"></div>
                <div class="elb-header" id="elb-header">
                    <div class="elb-header-left">
                        <span class="elb-header-title">📋 Live Event Log</span>
                        <span class="elb-header-badge" id="elb-badge" style="display:none;">0</span>
                    </div>
                    <div class="elb-header-right">
                        <span class="elb-conn-dot connecting" id="elb-conn-dot"></span>
                        <span class="elb-conn-text" id="elb-conn-text">Connecting…</span>
                        <span class="elb-toggle collapsed" id="elb-toggle">▾</span>
                    </div>
                </div>
                <div class="elb-body hidden" id="elb-body">
                    <div class="elb-toolbar">
                        <div class="elb-toolbar-left">
                            <span class="elb-info" id="elb-info">0 events</span>
                        </div>
                        <div class="elb-toolbar-right">
                            <button class="elb-clear-btn" id="elb-clear-btn">🗑️ Clear</button>
                        </div>
                    </div>
                    <div class="elb-list" id="elb-list">
                        <div class="elb-empty">No events yet. Actions in the Admin UI will appear here in real time.</div>
                    </div>
                </div>
            </div>
        `;

        const header = document.getElementById('elb-header');
        const body = document.getElementById('elb-body');
        const toggle = document.getElementById('elb-toggle');
        const list = document.getElementById('elb-list');
        const badge = document.getElementById('elb-badge');
        const info = document.getElementById('elb-info');
        const connDot = document.getElementById('elb-conn-dot');
        const connText = document.getElementById('elb-conn-text');
        const clearBtn = document.getElementById('elb-clear-btn');

        let eventSource = null;
        let pollTimer = null;
        let eventCount = 0;
        let expanded = false;

        // Add body padding to prevent overlap
        document.body.classList.add('elb-active');

        // ── Toggle expand/collapse ──
        header.addEventListener('click', function() {
            if (expanded) {
                collapse();
            } else {
                expand();
            }
        });

        function expand() {
            expanded = true;
            body.classList.remove('hidden');
            toggle.classList.remove('collapsed');
            document.body.classList.remove('elb-collapsed');
            connect();
        }

        function collapse() {
            expanded = false;
            body.classList.add('hidden');
            toggle.classList.add('collapsed');
            document.body.classList.add('elb-collapsed');
            disconnect();
        }

        // ── Connect to SSE with polling fallback ──
        function connect() {
            disconnect();

            // Load recent events first via REST
            loadRecentEvents();

            // Try SSE
            const params = new URLSearchParams();
            if (draftId) params.set('draft_id', draftId);
            const url = '/admin/api/events/live' + (params.toString() ? '?' + params.toString() : '');

            try {
                eventSource = new EventSource(url);

                eventSource.onopen = function() {
                    connDot.className = 'elb-conn-dot connected';
                    connText.textContent = 'Live (SSE)';
                    // Stop polling if SSE works
                    if (pollTimer) { clearInterval(pollTimer); pollTimer = null; }
                };

                eventSource.onmessage = function(e) {
                    try {
                        const event = JSON.parse(e.data);
                        renderEntry(event);
                    } catch(err) { /* ignore parse errors */ }
                };

                eventSource.onerror = function() {
                    connDot.className = 'elb-conn-dot disconnected';
                    connText.textContent = 'Reconnecting…';
                    eventSource.close();
                    eventSource = null;
                    // Fall back to polling
                    if (!pollTimer) {
                        pollTimer = setInterval(loadRecentEvents, POLL_INTERVAL);
                    }
                    // Retry SSE after delay
                    setTimeout(function() {
                        if (expanded && (!eventSource || eventSource.readyState === EventSource.CLOSED)) {
                            connect();
                        }
                    }, 3000);
                };
            } catch(e) {
                // EventSource not supported — polling only
                connDot.className = 'elb-conn-dot disconnected';
                connText.textContent = 'Polling (3s)';
                pollTimer = setInterval(loadRecentEvents, POLL_INTERVAL);
            }
        }

        function disconnect() {
            if (eventSource) {
                eventSource.close();
                eventSource = null;
            }
            if (pollTimer) {
                clearInterval(pollTimer);
                pollTimer = null;
            }
        }

        // ── Load recent events via REST ──
        async function loadRecentEvents() {
            const params = new URLSearchParams();
            params.set('limit', '50');
            if (draftId) params.set('draft_id', draftId);
            try {
                const resp = await fetch('/admin/api/events?' + params.toString());
                const data = await resp.json();
                if (data.events && data.events.length > 0) {
                    // Clear existing and render (events come newest-first from API)
                    list.innerHTML = '';
                    eventCount = 0;
                    // API returns newest first, we want newest at bottom
                    // So iterate in reverse
                    const sorted = data.events.slice().reverse();
                    sorted.forEach(function(e) { renderEntry(e, true); });
                }
            } catch(err) { /* silent */ }
        }

        // ── Render a single log entry ──
        function renderEntry(event, skipScroll) {
            // Remove empty placeholder
            const emptyEl = list.querySelector('.elb-empty');
            if (emptyEl) emptyEl.remove();

            const entry = document.createElement('div');
            entry.className = 'elb-entry';
            entry.setAttribute('data-event-id', event.id || '');

            const d = new Date(event.timestamp * 1000);
            const timeStr = d.toLocaleTimeString('de-DE', {
                hour: '2-digit', minute: '2-digit', second: '2-digit'
            });

            const hasDetail = event.detail && Object.keys(event.detail).length > 0;
            const detailId = 'elb-json-' + (event.id || Math.random().toString(36).substr(2, 9));

            const catClass = 'elb-cat-' + (event.category || 'system');
            const statusClass = 'elb-status-' + (event.status || 'info');

            entry.innerHTML =
                '<span class="elb-entry-time">' + escapeHtml(timeStr) + '</span>' +
                '<span class="elb-entry-cat ' + catClass + '">' + escapeHtml(event.category || 'system') + '</span>' +
                '<span class="elb-entry-status ' + statusClass + '">' + escapeHtml(event.status || 'info') + '</span>' +
                '<span class="elb-entry-msg">' + escapeHtml(event.message || event.action || '') + '</span>' +
                (hasDetail ? '<span class="elb-entry-detail" onclick="elbToggleJson(\'' + detailId + '\')">JSON</span>' : '');

            // Append to bottom (newest at bottom)
            list.appendChild(entry);

            if (hasDetail) {
                const jsonDiv = document.createElement('div');
                jsonDiv.className = 'elb-entry-json';
                jsonDiv.id = detailId;
                jsonDiv.textContent = JSON.stringify(event.detail, null, 2);
                list.appendChild(jsonDiv);
            }

            eventCount++;
            badge.textContent = eventCount;
            badge.style.display = 'inline-block';
            info.textContent = eventCount + ' events';

            // Auto-scroll to bottom (newest events at bottom)
            if (!skipScroll) {
                list.scrollTop = list.scrollHeight;
            }

            // Limit entries
            const entries = list.querySelectorAll('.elb-entry');
            if (entries.length > MAX_ENTRIES) {
                const toRemove = entries.length - MAX_ENTRIES;
                let removed = 0;
                for (let i = 0; i < toRemove; i++) {
                    const el = entries[i];
                    // Remove associated JSON div if present
                    const next = el.nextElementSibling;
                    if (next && next.classList.contains('elb-entry-json')) {
                        next.remove();
                    }
                    el.remove();
                    removed++;
                }
            }
        }

        // ── Clear button ──
        clearBtn.addEventListener('click', function(e) {
            e.stopPropagation();
            if (!confirm('Delete all events?')) return;
            fetch('/admin/api/events/clear', { method: 'POST' })
                .then(function(r) { return r.json(); })
                .then(function(data) {
                    list.innerHTML = '<div class="elb-empty">All events cleared.</div>';
                    eventCount = 0;
                    badge.style.display = 'none';
                    info.textContent = '0 events';
                })
                .catch(function() { /* silent */ });
        });

        // ── Toggle JSON detail ──
        window.elbToggleJson = function(id) {
            const el = document.getElementById(id);
            if (el) el.classList.toggle('show');
        };

        // ── Escape HTML helper ──
        function escapeHtml(str) {
            if (str == null) return '';
            const div = document.createElement('div');
            div.textContent = String(str);
            return div.innerHTML;
        }

        // ── Resize handle (drag to resize body height) ──
        const resizeHandle = document.getElementById('elb-resize-handle');
        const MIN_HEIGHT = 80;
        const MAX_HEIGHT = window.innerHeight - 120;

        // Restore saved height from localStorage
        try {
            const saved = localStorage.getItem('elb-body-height');
            if (saved) {
                const h = parseInt(saved, 10);
                if (h >= MIN_HEIGHT && h <= MAX_HEIGHT) {
                    document.documentElement.style.setProperty('--elb-body-height', h + 'px');
                }
            }
        } catch(e) {}

        resizeHandle.addEventListener('mousedown', function(e) {
            e.preventDefault();
            e.stopPropagation();

            const startY = e.clientY;
            const wrapper = document.getElementById('elb-wrapper');
            const currentHeight = parseInt(getComputedStyle(body).height, 10);

            wrapper.classList.add('resizing');
            resizeHandle.classList.add('dragging');
            document.body.style.cursor = 'ns-resize';
            document.body.style.userSelect = 'none';

            function onMouseMove(ev) {
                const delta = startY - ev.clientY; // up = positive = grow
                let newHeight = currentHeight + delta;
                const maxH = window.innerHeight - 120;
                if (newHeight < MIN_HEIGHT) newHeight = MIN_HEIGHT;
                if (newHeight > maxH) newHeight = maxH;
                document.documentElement.style.setProperty('--elb-body-height', newHeight + 'px');
            }

            function onMouseUp(ev) {
                document.removeEventListener('mousemove', onMouseMove);
                document.removeEventListener('mouseup', onMouseUp);
                wrapper.classList.remove('resizing');
                resizeHandle.classList.remove('dragging');
                document.body.style.cursor = '';
                document.body.style.userSelect = '';

                // Persist height
                const finalHeight = parseInt(getComputedStyle(body).height, 10);
                try {
                    localStorage.setItem('elb-body-height', String(finalHeight));
                } catch(e) {}
            }

            document.addEventListener('mousemove', onMouseMove);
            document.addEventListener('mouseup', onMouseUp);
        });

        // Auto-expand on first load if there's a draft_id (draft detail page)
        // On list pages, start collapsed
        if (draftId) {
            expand();
        } else {
            // Even when collapsed, connect to SSE so events are buffered
            // and ready to display when the user expands
            connect();
        }
    }

    // Initialize on DOMContentLoaded
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', initEventLogBar);
    } else {
        initEventLogBar();
    }
})();
