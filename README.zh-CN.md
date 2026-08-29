# Neuro-Symbolic Math Agent (神经符号数学 Agent)

> 面向数学教育题目生成、SymPy 门控校验与反思修正的神经符号 Agent 框架。

---

## 1. 项目概述 (Title & Description)

**Neuro-Symbolic Math Agent** 是一个面向数学教育评估的神经符号题目生成与校验系统。系统结合了大语言模型（LLM）的自然语言文本生成能力与 **SymPy 符号计算引擎** 的严格数学推演能力。

系统采用 **Verifier-Gated 控制器架构**：首先将输入的原始题目形式化解析为步骤依赖图（DAG），接着生成候选变式（同构变式 `isomorphic` 或单步支架题 `scaffold`），并送入 SymPy 验证器进行符号推演与求解校验。如果校验未通过，**反思引擎 (Reflection Engine)** 会将错误诊断分类转化为 Prompt 级别的修正约束，发起最多 $k$ 次重试闭环。若所有尝试均未通过，控制器将执行 **安全弃答机制**（`delivered = false`），绝不向学生交付未验证或数学上存在缺陷的题目。

---

## 2. 数据集信息 (Dataset Information)

项目包含面向数学教育评估的三大标准结构化数据集：

- **论文基准测试集 (`data/benchmark/`)**：
  - `gsm8k_200.jsonl` (200 题)：小学数学应用题、速率/时间、应用算术。
  - `math_200.jsonl` (200 题)：高中代数、二次方程、多项式方程组与定义域受限表达式。
  - `prm800k_200.jsonl` (200 题)：包含过程监督与步骤依赖标注的多步推导题。

- **全量规范化数据集 (`data/normalized/`)**：
  - `gsm8k.jsonl` (1,319 题)：全量规范化 GSM8K 测试集。
  - `math.jsonl` (1,187 题)：全量规范化 MATH 代数测试集。
  - `prm800k.jsonl` (4,500 题)：全量规范化 PRM800K Phase-2 测试集。

每条数据记录均包含 `dataset`、`record_id`、`problem` 题面文本、参考解答 `solution`、标准答案 `answer`、推导步骤 `steps` 与领域元数据 `metadata`。

---

## 3. 代码结构说明 (Code Information)

```text
release/
├── pyproject.toml                     # 项目标准打包配置文件 (pip install -e .)
├── config.json                        # API Base URL & Key 配置文件 (已在 git 忽略)
├── .env.example                       # 环境变量配置模版
├── README.md                          # 英文项目文档
├── README.zh-CN.md                    # 中文项目文档 (本文件)
├── src/                               # 核心引擎代码包
│   └── neuro_symbolic_math_agent/
│       ├── __init__.py
│       ├── agent.py                   # MathAgent 主控制器与 Verifier-Gated 闭环
│       ├── verifier.py                # SymPy 求解、实数根、定义域、CVI 与 SFS 检查器
│       ├── reflection.py              # 反思引擎 (错误代码 -> Prompt 约束映射)
│       ├── models.py                  # Problem, Step, Candidate, Verification 数据结构
│       ├── datasets.py                # 数据集下载、规范化清洗与抽样器
│       ├── providers.py               # OpenAI & SiliconFlow 标准库 API 客户端
│       ├── benchmark.py               # Oracle 与 LLM 基准评测引擎
│       ├── cli.py                     # 单题交互 CLI 终端入口
│       └── benchmark_cli.py           # Benchmark 评测 CLI 终端入口
├── web/                               # 响应式可视化 Web 应用
│   ├── server.py                      # FastAPI REST 后端 (/api/config, /api/datasets, /api/run)
│   ├── templates/
│   │   └── index.html                 # 响应式 Web 界面 (含 200 题浏览与流程图)
│   └── static/
│       ├── app.css                    # 前端 UI 样式与仪表盘
│       └── app.js                     # AJAX 交互与动态推演轨迹渲染脚本
├── examples/                          # 演示代码
│   ├── demo_run.py                    # 离线模拟与在线 API 交互演示脚本
│   └── quadratic_problem.json         # 结构化 JSON 输入示例
├── tests/                             # 单元测试
│   ├── test_agent.py                  # Verifier 校验、反思重试与安全弃答测试
│   └── test_datasets.py               # 数据集适配与 Oracle Benchmark 测试
├── scripts/                           # 实验与分析代码
│   ├── run_large_experiment.py        # 200 题三数据集 Benchmark 运行脚本
│   ├── run_small_api_benchmark.py     # 快速 API Benchmark 运行脚本
│   ├── analyze_experiment_for_paper.py# 统计分析工具 (Pass Rate, CVI, SFS, 置信区间)
│   └── audit_paper_results.py         # 论文实验数据可复算审计脚本
└── data/                              # 评测数据集
    └── benchmark/                     # 200 题基准 JSONL 数据集文件
```

---

## 4. 使用说明 (Usage Instructions)

### 1. 环境准备与安装

要求 Python 3.10 或更高版本。在项目根目录下执行：

```bash
# 创建并激活虚拟环境
python -m venv .venv
source .venv/bin/activate        # Linux/macOS
# .\.venv\Scripts\Activate.ps1   # Windows PowerShell

# 可编辑模式安装
pip install -e .
```

### 2. API 密钥配置 (`config.json`)

可以直接在项目根目录下的 [`config.json`](file:///d:/AI_DEV/ai_edu_8%E6%9C%88/ai_edu_math/release/config.json) 中显式填写 Key（代码会自动优先读取，无需在终端中设置环境变量）：

```json
{
  "siliconflow": {
    "api_key": "your-siliconflow-api-key-here",
    "base_url": "https://api.siliconflow.cn/v1",
    "model": "deepseek-ai/DeepSeek-V3"
  },
  "openai": {
    "api_key": "your-openai-api-key-here",
    "base_url": "https://api.openai.com/v1",
    "model": "gpt-4o-mini"
  }
}
```

### 3. 启动可视化交互 Web 应用

启动 FastAPI 后端服务浏览 200 题试题库并查看推演轨迹：

```bash
python -m uvicorn web.server:app --host 0.0.0.0 --port 8000
```

在浏览器打开 **`http://localhost:8000`** 即可：
- 检索与浏览 GSM8K、MATH、PRM800K 的 200 道试题。
- 实时配置 API Key/URL 或一键切换离线模拟模式。
- 可视化展示步骤 DAG、Attempt 尝试、SymPy 反思提示、CVI/SFS 得分及最终交付状态。

### 4. 运行 Python 演示脚本 (Demo Code)

```bash
python examples/demo_run.py
```

### 5. 运行测试与基准评测 (Tests & Benchmarks)

```bash
# 运行单元测试
python -m unittest discover -s tests

# 运行 300 次离线 Oracle 校验评测
math-agent-benchmark --mode oracle --max-attempts 2

# 重新生成固化测试集 (如 200 题)
math-agent-download --count 200 --seed 20260803
```

---

## 5. 环境要求 (Requirements)

项目依赖轻量级 Python 标准库与主流数据处理包：

- **Python**: $\ge 3.10$
- **SymPy**: $\ge 1.12$（符号数学解析、求解与定义域校验）
- **FastAPI**: $\ge 0.100.0$（Web REST API 服务）
- **Uvicorn**: $\ge 0.22.0$（ASGI 服务器）
- **Pydantic**: $\ge 2.0.0$（数据验证）
- **Jinja2**: $\ge 3.0.0$（模板渲染）
- **Requests**: $\ge 2.28.0$（HTTP 下载客户端）

---

## 6. 方法论 (Methodology)

系统遵循四阶段的神经符号流水线：

1. **形式化解析 (Formalization/Decomposer)**：将自然语言题目解析为包含变量、SymPy 语法方程、步骤依赖树 (`Step`) 和定义域约束 (`DomainConstraint`) 的结构化 `Problem` 对象。
2. **题目生成 (Item Generation)**：
   - **`isomorphic` (同构模式)**：保持方程数量、次数、运算符复杂度与步骤逻辑一致，仅替换情境与数值。
   - **`scaffold` (支架模式)**：围绕指定的卡点步骤 (`target_step_id`) 生成降低维度的过渡过渡题。
3. **符号校验 (`SymbolicVerifier`)**：
   - **可解性与实数根**：计算 SymPy 精确解并校验有限实数解。
   - **定义域约束**：校验解集是否满足不等式（如 $x > 0$、$x \neq 0$）。
   - **Clean-Value Index (CVI)**：评估数值教学适宜度（偏好简单整数和分数，惩罚复杂小数和根式）。
   - **Symbolic Feature-Similarity (SFS)**：评估结构同构度匹配（$\ge 0.60$ 阈值）。
4. **反思修正与安全弃答 (Reflective Remediation & Abstention)**：校验失败时，`ReflectionEngine` 将错误代码（`NO_SOLUTION_ERROR`、`DOMAIN_VIOLATION_ERROR` 等）映射为自然语言 Prompt 约束进行最多 $k$ 次重试；若全部失败则设置 `delivered = false` 拒绝交付。

---

## 7. 引用规范 (Citations)

如果在研究或学术论文中使用本项目或数据集，请引用：

```bibtex
@article{neuro_symbolic_math_agent_2026,
  title={Verifier-Gated Neuro-Symbolic Agent for Pedagogical Mathematics Item Generation and Reflective Remediation},
  author={Antigravity Team},
  journal={PeerJ Computer Science Draft},
  year={2026}
}
```

同时请引用各原始开源数据集论文：

- **GSM8K**: Cobbe et al., 2021. *Training Verifiers to Solve Math Word Problems*. [arXiv:2110.14168](https://arxiv.org/abs/2110.14168).
- **MATH**: Hendrycks et al., 2021. *Measuring Mathematical Problem Solving With the MATH Dataset*. [arXiv:2103.03874](https://arxiv.org/abs/2103.03874).
- **PRM800K**: Lightman et al., 2023. *Let's Verify Step by Step*. [arXiv:2305.20050](https://arxiv.org/abs/2305.20050).

---

## 8. 许可证与贡献指南 (License & Contribution Guidelines)

### 开源许可证
本项目开源代码使用 **MIT License**。随附数据集（GSM8K, MATH, PRM800K）遵循各自原始开源协议。

### 贡献指南
1. **Pull Requests**：欢迎提交 Pull Request！请确保所有新功能或修改均包含对应的单元测试 (`tests/`)。
2. **校验规则扩展**：在添加新的 SymPy 校验或 CVI 指标时，请保持与现有 `Problem` 数据结构向下兼容。
3. **安全规范**：**切勿将真实 API 密钥或包含敏感信息的配置文件提交到 Git 仓库**。请使用 `config.json` 或 `.env.example` 作为范本。
