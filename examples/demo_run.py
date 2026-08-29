"""
Neuro-Symbolic Math Agent Demo Script
示范 Neuro-Symbolic Math Agent 的离线模拟运行与在线 API 交互生成
"""

import os
import sys
from pathlib import Path

# 将 src 目录添加到 sys.path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from neuro_symbolic_math_agent.agent import MathAgent
from neuro_symbolic_math_agent.models import Candidate, DomainConstraint, GenerationMode, Problem, Step
from neuro_symbolic_math_agent.providers import (
    LLMGenerator,
    OpenAICompatibleClient,
    SequenceGenerator,
    _load_config_file,
)


def get_sample_problem() -> Problem:
    """构造示例一二次方程题目与步骤依赖图"""
    return Problem(
        problem_id="quadratic_demo_1",
        text="Solve for x: x^2 - 5*x + 6 = 0 given x > 0.",
        variables=["x"],
        equations=["x**2 - 5*x + 6 = 0"],
        domain_constraints=[DomainConstraint("x", "x > 0")],
        steps=[
            Step(
                step_id="s1",
                name="Factorization",
                description="Factor the quadratic equation into (x-2)(x-3) = 0",
                equations=["(x-2)*(x-3) = 0"],
                dependencies=[],
            ),
            Step(
                step_id="s2",
                name="Root extraction",
                description="Solve for roots x = 2 or x = 3",
                equations=["x = 2", "x = 3"],
                dependencies=["s1"],
            ),
        ],
    )


def run_offline_demo():
    print("\n" + "=" * 65)
    print(" 1. 离线模拟 Demo (Offline Mode - 无需 API Key)")
    print("=" * 65)

    problem = get_sample_problem()

    # 构造预设的序列候选题目 (模拟 第一次失败 -> 第二次反思纠正成功)
    candidate_attempt_1 = Candidate(
        question="Find positive x such that x^2 + 4 = 0.",
        variables=["x"],
        equations=["x**2 + 4 = 0"],  # 无实根，触发 SymPy Error
        answer="x = 2",
    )
    candidate_attempt_2 = Candidate(
        question="Find positive x such that x^2 - 7*x + 12 = 0.",
        variables=["x"],
        equations=["x**2 - 7*x + 12 = 0"],  # 有有效整数根 x=3, 4
        answer="x = 3, 4",
    )

    generator = SequenceGenerator([candidate_attempt_1, candidate_attempt_2])
    agent = MathAgent(generator=generator, max_attempts=3)

    print("\n[运行: Isomorphic Mode (同构变式生成) 与 反思重试闭环]")
    result = agent.run(problem, mode=GenerationMode.ISOMORPHIC)

    print(f"-> 交付成功状态 (delivered): {result.delivered}")
    print(f"-> 尝试总次数 (attempts_used): {len(result.attempts)}")
    if result.candidate and result.verification:
        print(f"\n[通过 SymPy 门控验证的最终生成题目]")
        print(f"   题目文本: {result.candidate.question}")
        print(f"   方程系统: {result.candidate.equations}")
        print(f"   SymPy 求解解集: {result.verification.solutions}")
        print(f"   Clean-Value Index (CVI Score): {result.verification.cvi:.4f}")
        print(f"   Symbolic Feature-Similarity (SFS Score): {result.verification.isomorphism_score:.4f}")
        print(f"   最终判定 accepted: {result.verification.accepted}")


def run_online_demo():
    print("\n" + "=" * 65)
    print(" 2. 在线大模型交互 Demo (Online LLM Mode)")
    print("=" * 65)

    cfg = _load_config_file()
    openai_cfg = cfg.get("openai", {}) if isinstance(cfg.get("openai"), dict) else {}

    default_key = os.getenv("OPENAI_API_KEY") or openai_cfg.get("api_key", "")
    default_url = os.getenv("OPENAI_BASE_URL") or openai_cfg.get("base_url", "https://api.openai.com/v1")
    default_model = os.getenv("OPENAI_MODEL") or openai_cfg.get("model", "gpt-4o-mini")

    try:
        print("\n请输入大模型 API 配置（直接回车使用默认值/配置文件值）：")
        
        # 交互式输入
        api_url_input = input(f"-> API Base URL [{default_url}]: ").strip()
        api_url = api_url_input if api_url_input else default_url

        model_input = input(f"-> 模型名称 [{default_model}]: ").strip()
        model_name = model_input if model_input else default_model

        if default_key:
            masked_key = default_key[:6] + "..." + default_key[-4:] if len(default_key) > 10 else "***"
            api_key_input = input(f"-> API Key [{masked_key}]: ").strip()
            api_key = api_key_input if api_key_input else default_key
        else:
            api_key = input("-> API Key: ").strip()

    except EOFError:
        api_url = default_url
        model_name = default_model
        api_key = default_key

    if not api_key:
        print("\n[提示] 未提供 API Key，跳过在线大模型测试。可以在 config.json 或环境变量中配置 API Key。")
        return

    print(f"\n[正在连接大模型 API: {api_url} | 模型: {model_name}]...")

    try:
        client = OpenAICompatibleClient(base_url=api_url, api_key=api_key, model=model_name)
        generator = LLMGenerator(client)
        agent = MathAgent(generator=generator, max_attempts=3)

        problem = get_sample_problem()

        print("\n[正在在线请求大模型生成同构变式题目并经 SymPy 检验...]")
        result = agent.run(problem, mode=GenerationMode.ISOMORPHIC)

        print(f"\n-> 交付成功状态 (delivered): {result.delivered}")
        print(f"-> 尝试总次数 (attempts_used): {len(result.attempts)}")
        if result.candidate and result.verification:
            print(f"\n[大模型在线实时生成并通过 SymPy 验证的题目]")
            print(f"   题目文本: {result.candidate.question}")
            print(f"   方程系统: {result.candidate.equations}")
            print(f"   SymPy 求解解集: {result.verification.solutions}")
            print(f"   Clean-Value Index (CVI Score): {result.verification.cvi:.4f}")
            print(f"   Symbolic Feature-Similarity (SFS Score): {result.verification.isomorphism_score:.4f}")
            print(f"   最终判定 accepted: {result.verification.accepted}")
        else:
            print(f"\n[弃答说明] {result.abstention_reason}")

    except Exception as exc:
        print(f"\n[在线请求失败] {type(exc).__name__}: {exc}")


def main():
    print("=" * 65)
    print("   Neuro-Symbolic Math Agent 双模式 Demo 演示")
    print("=" * 65)

    # 1. 先进行离线 Demo 演示
    run_offline_demo()

    # 2. 询问是否进行在线 LLM 真实 API 测试
    if len(sys.argv) > 1 and sys.argv[1] in ("--online", "-o"):
        run_online_demo()
    elif sys.stdin.isatty():
        try:
            choice = input("\n是否想要测试在线大模型生成？(y/N): ").strip().lower()
            if choice in ("y", "yes"):
                run_online_demo()
            else:
                print("\n已完成离线 Demo 演示。感谢使用！")
        except EOFError:
            print("\n已完成离线 Demo 演示。感谢使用！")
    else:
        print("\n非交互终端环境，跳过在线测试。在终端附加 `--online` 参数可开启在线测试。")


if __name__ == "__main__":
    main()
