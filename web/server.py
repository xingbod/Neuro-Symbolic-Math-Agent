"""
Neuro-Symbolic Math Agent Interactive Web Server
FastAPI Web 后端，提供 API 设置、200 题基准库加载、实时推演与全流程可视化接口
"""

import json
import os
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

# 将 src 目录添加到 sys.path
BASE_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(BASE_DIR / "src"))

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from neuro_symbolic_math_agent.agent import MathAgent
from neuro_symbolic_math_agent.datasets import BenchmarkRecord, load_benchmark
from neuro_symbolic_math_agent.models import Candidate, DomainConstraint, GenerationMode, Problem, Step
from neuro_symbolic_math_agent.providers import (
    LLMGenerator,
    OpenAICompatibleClient,
    SequenceGenerator,
    _load_config_file,
)

app = FastAPI(title="Neuro-Symbolic Math Agent Interactive UI")

# 静态目录与模板挂载
WEB_DIR = Path(__file__).parent
app.mount("/static", StaticFiles(directory=str(WEB_DIR / "static")), name="static")
templates = Jinja2Templates(directory=str(WEB_DIR / "templates"))


def get_data_benchmark_dir() -> Path:
    """获取 200 题基准数据目录"""
    bm_dir = BASE_DIR / "data" / "benchmark"
    if bm_dir.exists():
        return bm_dir
    return BASE_DIR / "data" / "normalized"


def mask_key(key: str) -> str:
    if not key or len(key) < 8:
        return ""
    return key[:4] + "..." + key[-4:]


class RunRequest(BaseModel):
    provider: str = "openai"
    base_url: str | None = None
    api_key: str | None = None
    model: str | None = None
    use_offline: bool = False
    dataset: str = "gsm8k"
    record_id: str | None = None
    custom_problem_text: str | None = None
    mode: str = "isomorphic"
    target_step_id: str | None = None


@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    """渲染 Web 主页"""
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/api/config")
def get_config():
    """获取配置文件默认值 (隐藏实际 Key 细节)"""
    cfg = _load_config_file()
    openai_cfg = cfg.get("openai", {}) if isinstance(cfg.get("openai"), dict) else {}
    siliconflow_cfg = cfg.get("siliconflow", {}) if isinstance(cfg.get("siliconflow"), dict) else {}

    openai_key = os.getenv("OPENAI_API_KEY") or openai_cfg.get("api_key", "")
    siliconflow_key = os.getenv("SILICONFLOW_API_KEY") or siliconflow_cfg.get("api_key", "")

    return {
        "openai": {
            "base_url": os.getenv("OPENAI_BASE_URL") or openai_cfg.get("base_url", "https://api.openai.com/v1"),
            "model": os.getenv("OPENAI_MODEL") or openai_cfg.get("model", "gpt-4o-mini"),
            "has_key": bool(openai_key),
            "masked_key": mask_key(openai_key),
        },
        "siliconflow": {
            "base_url": os.getenv("SILICONFLOW_BASE_URL") or siliconflow_cfg.get("base_url", "https://api.siliconflow.cn/v1"),
            "model": os.getenv("SILICONFLOW_MODEL") or siliconflow_cfg.get("model", "deepseek-ai/DeepSeek-V3"),
            "has_key": bool(siliconflow_key),
            "masked_key": mask_key(siliconflow_key),
        },
    }


@app.get("/api/datasets")
def list_datasets(dataset: str = "gsm8k"):
    """加载 200 题基准数据列表"""
    data_dir = get_data_benchmark_dir()
    
    # 匹配 _200.jsonl 或 _50.jsonl 或 .jsonl
    target_file = None
    for candidate_name in (f"{dataset}_200.jsonl", f"{dataset}_50.jsonl", f"{dataset}.jsonl"):
        path = data_dir / candidate_name
        if path.exists():
            target_file = path
            break

    if not target_file:
        raise HTTPException(status_code=404, detail=f"Dataset {dataset} not found in {data_dir}")

    records = load_benchmark(target_file)
    result_list = []
    for r in records:
        result_list.append({
            "dataset": r.dataset,
            "record_id": r.record_id,
            "problem": r.problem,
            "answer": r.answer,
            "solution": r.solution,
            "steps": r.steps,
        })
    return {"dataset": dataset, "count": len(result_list), "records": result_list}


def record_to_problem(record: BenchmarkRecord) -> Problem:
    """将基准记录转化为 Agent 所需的 Problem 结构体"""
    return Problem(
        problem_id=record.record_id,
        text=record.problem,
        variables=["x"],
        equations=["x = " + (record.answer if record.answer and record.answer.replace(".", "").isdigit() else "1")],
        domain_constraints=[DomainConstraint("x", "x > 0")],
        steps=[
            Step(
                step_id=f"s{idx + 1}",
                name=f"step_{idx + 1}",
                description=step_text,
                equations=[],
                dependencies=[f"s{idx}"] if idx > 0 else [],
            )
            for idx, step_text in enumerate(record.steps[:5])
        ] or [Step("s1", "solve", "Solve the problem", [], [])],
    )


@app.post("/api/run")
def run_agent(req: RunRequest):
    """运行 Agent 并返回可视化推演轨迹"""
    # 1. 查找/准备 Problem
    problem = None
    if req.custom_problem_text:
        problem = Problem(
            problem_id="custom_1",
            text=req.custom_problem_text,
            variables=["x"],
            equations=["x**2 - 5*x + 6 = 0"],
            domain_constraints=[DomainConstraint("x", "x > 0")],
            steps=[
                Step("s1", "Factor", "Factor into (x-2)(x-3)=0", ["(x-2)*(x-3)=0"], []),
                Step("s2", "Roots", "Roots x=2, x=3", ["x=2", "x=3"], ["s1"]),
            ],
        )
    elif req.record_id:
        data_dir = get_data_benchmark_dir()
        target_file = None
        for candidate_name in (f"{req.dataset}_200.jsonl", f"{req.dataset}_50.jsonl", f"{req.dataset}.jsonl"):
            path = data_dir / candidate_name
            if path.exists():
                target_file = path
                break
        if target_file:
            records = load_benchmark(target_file)
            matched = [r for r in records if r.record_id == req.record_id]
            if matched:
                problem = record_to_problem(matched[0])

    if not problem:
        # 默认回退示例题目
        problem = Problem(
            problem_id="quadratic_demo_1",
            text="Solve for x: x^2 - 5*x + 6 = 0 given x > 0.",
            variables=["x"],
            equations=["x**2 - 5*x + 6 = 0"],
            domain_constraints=[DomainConstraint("x", "x > 0")],
            steps=[
                Step("s1", "Factorization", "Factor into (x-2)(x-3)=0", ["(x-2)*(x-3)=0"], []),
                Step("s2", "Roots", "Roots x=2, x=3", ["x=2", "x=3"], ["s1"]),
            ],
        )

    # 2. 配置 Generator (离线规则 或 在线大模型 API)
    if req.use_offline:
        c1 = Candidate(
            question="Find positive x such that x^2 + 4 = 0.",
            variables=["x"],
            equations=["x**2 + 4 = 0"],
            answer="x = 2",
        )
        c2 = Candidate(
            question="Find positive x such that x^2 - 7*x + 12 = 0.",
            variables=["x"],
            equations=["x**2 - 7*x + 12 = 0"],
            answer="x = 3, 4",
        )
        generator = SequenceGenerator([c1, c2])
    else:
        # 获取 Key 与 Base URL
        cfg = _load_config_file()
        prov_cfg = cfg.get(req.provider, {}) if isinstance(cfg.get(req.provider), dict) else {}
        
        api_key = req.api_key or os.getenv(f"{req.provider.upper()}_API_KEY") or prov_cfg.get("api_key")
        if not api_key:
            raise HTTPException(status_code=400, detail=f"API Key for provider '{req.provider}' is missing. Please enter your API Key.")

        base_url = req.base_url or os.getenv(f"{req.provider.upper()}_BASE_URL") or prov_cfg.get("base_url")
        model = req.model or os.getenv(f"{req.provider.upper()}_MODEL") or prov_cfg.get("model")

        if req.provider == "openai":
            base_url = base_url or "https://api.openai.com/v1"
            model = model or "gpt-4o-mini"
        elif req.provider == "siliconflow":
            base_url = base_url or "https://api.siliconflow.cn/v1"
            model = model or "deepseek-ai/DeepSeek-V3"

        try:
            client = OpenAICompatibleClient(base_url=base_url, api_key=api_key, model=model)
            generator = LLMGenerator(client)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Failed to initialize client: {exc}")

    # 3. 运行 MathAgent 并组装可视化轨迹
    mode_enum = GenerationMode.SCAFFOLD if req.mode == "scaffold" else GenerationMode.ISOMORPHIC
    target_step = req.target_step_id if mode_enum == GenerationMode.SCAFFOLD else None
    if mode_enum == GenerationMode.SCAFFOLD and not target_step and problem.steps:
        target_step = problem.steps[0].step_id

    agent = MathAgent(generator=generator, max_attempts=3)
    result = agent.run(problem, mode=mode_enum, target_step_id=target_step)

    # 4. 转换可视化数据
    attempts_trace = []
    for att in result.attempts:
        candidate_data = att.candidate.to_dict() if att.candidate else None
        verification_data = None
        if att.verification:
            verification_data = {
                "accepted": att.verification.accepted,
                "solutions": att.verification.solutions,
                "cvi": round(att.verification.cvi, 4),
                "isomorphism_score": round(att.verification.isomorphism_score, 4),
                "issues": [asdict(issue) for issue in att.verification.issues],
            }
        attempts_trace.append({
            "number": att.number,
            "candidate": candidate_data,
            "verification": verification_data,
            "reflection": att.reflection,
            "error": att.generation_error,
        })

    final_verification = None
    if result.verification:
        final_verification = {
            "accepted": result.verification.accepted,
            "solutions": result.verification.solutions,
            "cvi": round(result.verification.cvi, 4),
            "isomorphism_score": round(result.verification.isomorphism_score, 4),
            "issues": [asdict(issue) for issue in result.verification.issues],
        }

    return {
        "problem": {
            "problem_id": problem.problem_id,
            "text": problem.text,
            "variables": problem.variables,
            "equations": problem.equations,
            "steps": [asdict(s) for s in problem.steps],
        },
        "mode": req.mode,
        "delivered": result.delivered,
        "attempts_used": len(result.attempts),
        "attempts": attempts_trace,
        "final_candidate": result.candidate.to_dict() if result.candidate else None,
        "final_verification": final_verification,
        "abstention_reason": result.abstention_reason,
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("web.server:app", host="0.0.0.0", port=8000, reload=True)
