// 全局状态
let currentDataset = "gsm8k";
let datasetRecords = [];
let selectedRecord = null;
let currentConfig = {};

document.addEventListener("DOMContentLoaded", () => {
    loadConfig();
    switchDataset("gsm8k");
});

// 加载默认配置
async function loadConfig() {
    try {
        const res = await fetch("/api/config");
        currentConfig = await res.json();
        onProviderChange();
    } catch (e) {
        console.error("加载配置失败:", e);
    }
}

// 切换服务商触发默认 URL & Model 填充
function onProviderChange() {
    const provider = document.getElementById("config-provider").value;
    const info = currentConfig[provider] || {};
    document.getElementById("config-base-url").value = info.base_url || "";
    document.getElementById("config-model").value = info.model || "";
    document.getElementById("config-api-key").placeholder = info.has_key ? `已配置环境变量 (${info.masked_key})` : "填入 API 密钥";
}

// 离线模式切换
function toggleOfflineMode() {
    const isOffline = document.getElementById("config-offline").checked;
    const inputs = ["config-provider", "config-base-url", "config-api-key", "config-model"];
    inputs.forEach(id => {
        document.getElementById(id).disabled = isOffline;
    });
}

// 切换 200 题数据集 (gsm8k, math, prm800k)
async function switchDataset(name) {
    currentDataset = name;
    document.querySelectorAll(".btn-dataset").forEach(btn => {
        btn.classList.toggle("active", btn.dataset.ds === name);
    });

    const listEl = document.getElementById("problem-list");
    listEl.innerHTML = `<div class="loading-spinner"><i class="fa-solid fa-spinner fa-spin"></i> 正在加载 ${name.toUpperCase()} (200 题) 数据集...</div>`;

    try {
        const res = await fetch(`/api/datasets?dataset=${name}`);
        const data = await res.json();
        datasetRecords = data.records || [];
        document.getElementById("dataset-count-badge").innerText = `${datasetRecords.length} 题`;
        renderProblemList(datasetRecords);

        // 默认选中第一题
        if (datasetRecords.length > 0) {
            selectProblem(datasetRecords[0].record_id);
        }
    } catch (e) {
        listEl.innerHTML = `<div class="loading-spinner" style="color:red;">加载数据集失败: ${e.message}</div>`;
    }
}

// 渲染 200 题列表
function renderProblemList(records) {
    const listEl = document.getElementById("problem-list");
    if (records.length === 0) {
        listEl.innerHTML = `<div class="loading-spinner">未找到匹配的题目</div>`;
        return;
    }

    listEl.innerHTML = records.map(r => `
        <div class="problem-item ${selectedRecord && selectedRecord.record_id === r.record_id ? 'selected' : ''}" 
             id="item-${r.record_id}" onclick="selectProblem('${r.record_id}')">
            <div class="item-id"><i class="fa-solid fa-hashtag"></i> ${r.record_id}</div>
            <div class="item-text">${escapeHtml(r.problem)}</div>
        </div>
    `).join("");
}

// 过滤搜索
function filterProblemList() {
    const query = document.getElementById("search-input").value.toLowerCase().strip();
    if (!query) {
        renderProblemList(datasetRecords);
        return;
    }
    const filtered = datasetRecords.filter(r => 
        r.record_id.toLowerCase().includes(query) || r.problem.toLowerCase().includes(query)
    );
    renderProblemList(filtered);
}

// 选中某一题
function selectProblem(recordId) {
    selectedRecord = datasetRecords.find(r => r.record_id === recordId);
    if (!selectedRecord) return;

    document.querySelectorAll(".problem-item").forEach(el => el.classList.remove("selected"));
    const selectedEl = document.getElementById(`item-${recordId}`);
    if (selectedEl) selectedEl.classList.add("selected");

    document.getElementById("selected-id-tag").innerText = selectedRecord.record_id;
    document.getElementById("selected-problem-text").innerText = selectedRecord.problem;
    document.getElementById("selected-solution-text").innerText = `解法/步骤参考: ${selectedRecord.solution || '暂无'}`;

    // 更新 Scaffold 目标步骤下拉菜单
    const stepSelect = document.getElementById("target-step-id");
    stepSelect.innerHTML = "";
    const steps = selectedRecord.steps || ["step_1"];
    steps.forEach((step, idx) => {
        const opt = document.createElement("option");
        opt.value = `s${idx + 1}`;
        opt.innerText = `Step ${idx + 1}: ${step.substring(0, 40)}...`;
        stepSelect.appendChild(opt);
    });
}

// 切换同构/支架模式选项
function toggleModeOptions() {
    const mode = document.querySelector('input[name="gen-mode"]:checked').value;
    document.getElementById("scaffold-options").style.display = mode === "scaffold" ? "block" : "none";
}

// 核心：运行 Agent 流程并渲染可视化轨迹
async function runAgentPipeline() {
    if (!selectedRecord) {
        alert("请先在左侧选择一道题目！");
        return;
    }

    const btn = document.getElementById("btn-run");
    btn.disabled = true;
    btn.innerHTML = `<i class="fa-solid fa-spinner fa-spin"></i> 正在推演与 SymPy 门控检验中...`;

    document.getElementById("empty-state").style.display = "none";
    document.getElementById("execution-trace-container").style.display = "block";

    const payload = {
        provider: document.getElementById("config-provider").value,
        base_url: document.getElementById("config-base-url").value,
        api_key: document.getElementById("config-api-key").value,
        model: document.getElementById("config-model").value,
        use_offline: document.getElementById("config-offline").checked,
        dataset: currentDataset,
        record_id: selectedRecord.record_id,
        mode: document.querySelector('input[name="gen-mode"]:checked').value,
        target_step_id: document.getElementById("target-step-id").value,
    };

    try {
        const res = await fetch("/api/run", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload)
        });

        if (!res.ok) {
            const errData = await res.json();
            throw new Error(errData.detail || "执行失败");
        }

        const data = await res.json();
        renderVisualization(data);
    } catch (e) {
        alert(`运行 Agent 失败: ${e.message}`);
    } finally {
        btn.disabled = false;
        btn.innerHTML = `<i class="fa-solid fa-bolt"></i> 运行 Neuro-Symbolic Agent 流程`;
    }
}

// 渲染可视化轨迹卡片
function renderVisualization(data) {
    // 阶段 1: Decomposer 结构
    document.getElementById("decomp-vars").innerText = (data.problem.variables || []).join(", ");
    document.getElementById("decomp-eqs").innerText = (data.problem.equations || []).join("; ");

    const dagContainer = document.getElementById("decomp-dag-tree");
    dagContainer.innerHTML = (data.problem.steps || []).map(s => `
        <div class="dag-node">
            <strong>[${s.step_id}] ${escapeHtml(s.name)}:</strong> ${escapeHtml(s.description)}
            ${s.dependencies && s.dependencies.length ? `<span style="color:#64748b;">(依赖: ${s.dependencies.join(",")})</span>` : ''}
        </div>
    `).join("");

    // 阶段 2: 尝试与反思重试列表
    const attemptsContainer = document.getElementById("attempts-list-container");
    attemptsContainer.innerHTML = (data.attempts || []).map(att => {
        const cand = att.candidate || {};
        const verif = att.verification || {};
        const isAccepted = verif.accepted;

        return `
            <div class="attempt-card">
                <div class="attempt-header">
                    <span><i class="fa-solid fa-code-commit"></i> Attempt #${att.number}</span>
                    <span class="${isAccepted ? 'status-accepted' : 'status-rejected'}">
                        ${isAccepted ? '<i class="fa-solid fa-circle-check"></i> Accepted (通过检验)' : '<i class="fa-solid fa-circle-xmark"></i> Rejected (检验拒绝)'}
                    </span>
                </div>
                <div style="font-size: 12px; color: #334155;">
                    <strong>生成题面:</strong> ${escapeHtml(cand.question || 'N/A')}
                </div>
                <div style="font-size: 11px; font-family: monospace; color: #475569;">
                    <strong>方程系统:</strong> ${JSON.stringify(cand.equations || [])}
                </div>
                ${verif.issues && verif.issues.length ? `
                    <div style="font-size: 11px; color: #dc2626;">
                        <strong>SymPy 发现的缺陷 (Issues):</strong> ${verif.issues.map(i => `${i.code}: ${i.message}`).join("; ")}
                    </div>
                ` : ''}
                ${att.reflection ? `
                    <div class="reflection-alert">
                        <i class="fa-solid fa-lightbulb"></i> <strong>反思引擎指令 (Reflection Prompt):</strong> ${escapeHtml(att.reflection)}
                    </div>
                ` : ''}
            </div>
        `;
    }).join("");

    // 阶段 3: 最终结果判定
    const banner = document.getElementById("result-status-banner");
    const isDelivered = data.delivered;

    banner.className = `result-status-banner ${isDelivered ? 'banner-success' : 'banner-fail'}`;
    document.getElementById("status-title").innerText = isDelivered ? "DELIVERED: TRUE" : "DELIVERED: FALSE (安全弃答)";
    document.getElementById("status-sub").innerText = isDelivered 
        ? "候选题目通过了 SymPy 的实数解、定义域约束、CVI 分数及 SFS 同构判定，安全交付给学生。" 
        : `多次重试均未通过验证。弃答原因: ${data.abstention_reason || '所有候选方程均未能满足检验要求'}`;

    const finalVerif = data.final_verification || {};
    document.getElementById("metric-cvi").innerText = finalVerif.cvi !== undefined ? finalVerif.cvi.toFixed(4) : "N/A";
    document.getElementById("metric-sfs").innerText = finalVerif.isomorphism_score !== undefined ? finalVerif.isomorphism_score.toFixed(4) : "N/A";
    document.getElementById("metric-solutions").innerText = finalVerif.solutions ? JSON.stringify(finalVerif.solutions) : "[]";

    const finalBox = document.getElementById("final-question-box");
    if (isDelivered && data.final_candidate) {
        finalBox.style.display = "block";
        document.getElementById("final-question-text").innerText = data.final_candidate.question;
        document.getElementById("final-question-eqs").innerText = `候选方程: ${JSON.stringify(data.final_candidate.equations)}`;
    } else {
        finalBox.style.display = "none";
    }
}

function escapeHtml(text) {
    if (!text) return "";
    return text.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
}
