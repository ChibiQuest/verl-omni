import ast
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


def _parse(relative_path: str) -> ast.Module:
    return ast.parse((REPO_ROOT / relative_path).read_text(encoding="utf-8"))


def test_omni_fsdp_turbo_engine_registration_and_mro():
    tree = _parse("verl_omni/workers/engine/fsdp_turbo/omni_impl.py")
    engine = next(node for node in tree.body if isinstance(node, ast.ClassDef))
    assert engine.name == "OmniFSDPTurboEngine"
    assert [base.id for base in engine.bases] == ["FSDPTurboEngineWithLMHead", "OmniFSDPEngine"]

    decorator = engine.decorator_list[0]
    keywords = {item.arg: ast.literal_eval(item.value) for item in decorator.keywords}
    assert keywords == {"model_type": "omni_model", "backend": "fsdp_turbo", "device": ["cuda", "npu"]}


def test_qwen35_moe_adapter_registration():
    tree = _parse("verl_omni/pipelines/qwen3_5/training_adapter.py")
    adapter = next(node for node in tree.body if isinstance(node, ast.ClassDef))
    registrations = {
        (ast.literal_eval(decorator.args[0]), ast.literal_eval(decorator.keywords[0].value))
        for decorator in adapter.decorator_list
    }
    assert registrations == {
        ("Qwen3_5MoeForConditionalGeneration", "all"),
        ("Qwen3_5MoeForConditionalGeneration", "thinker"),
    }
