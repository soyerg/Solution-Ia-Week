/* =================================================================
   HISTORY.JS — Shared history sidebar used by both Conveyor & Similarity
   ================================================================= */

const AppHistory = (() => {
    // ---------- state ----------
    const items = [];          // { id, file, blobUrl, name, result, timestamp }
    let nextId = 1;
    let onSelect = null;       // callback(item) when user clicks an item

    // ---------- DOM ----------
    const list = () => document.getElementById("sharedHistoryList");

    // ---------- public API ----------

    /** Register a callback for when user clicks a history entry */
    function setOnSelect(cb) { onSelect = cb; }

    /** Add an analysed image to the shared history */
    function add(file, blobUrl, name, result) {
        const item = {
            id: nextId++,
            file,            // original File object (for re-upload to /api/similar)
            blobUrl,         // object-URL for thumbnail / preview
            name,
            result,          // { label, label_fr, confidence, inference_time_ms }
            timestamp: new Date(),
        };
        items.unshift(item); // newest first
        render();
        return item;
    }

    /** Get all history items (newest first) */
    function getAll() { return items; }

    /** Find by id */
    function getById(id) { return items.find(i => i.id === id); }

    /** Clear everything */
    function clear() {
        items.length = 0;
        nextId = 1;
        render();
    }

    /** Render the sidebar list */
    function render() {
        const el = list();
        if (!el) return;

        if (items.length === 0) {
            el.innerHTML = '<div class="history-empty">Aucune pièce analysée</div>';
            return;
        }

        el.innerHTML = "";
        items.forEach((item) => {
            const isOk = item.result && item.result.label === "ok";
            const div = document.createElement("div");
            div.className = "history-item";
            div.dataset.historyId = item.id;
            div.innerHTML = `
                <img class="history-thumb" src="${item.blobUrl}" alt="" />
                <div class="history-info">
                    <div class="history-name">${item.name}</div>
                    <div class="history-result ${isOk ? "ok" : "def"}">${
                        item.result ? item.result.label_fr : "—"
                    }</div>
                </div>
                <div class="history-actions">
                    <button class="history-sim-btn" title="Rechercher les similaires">🔍</button>
                </div>
            `;

            // Click anywhere on the row → select for similarity
            div.addEventListener("click", () => {
                if (onSelect) onSelect(item);
            });

            el.appendChild(div);
        });
    }

    return { add, getAll, getById, clear, render, setOnSelect };
})();
