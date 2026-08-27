"""Basic single-video evaluation example."""
from video_eval.core.registry import initialize_registries
from video_eval.core.device import DeviceManager
from video_eval.core.pipeline import Pipeline
from video_eval.core.schemas import ProductInfo


def main():
    initialize_registries()

    config = {
        "evaluators": {
            "technical_quality": {"enabled": True},
            "vlm_judge": {"enabled": True, "backend": "mock"},
            "compliance": {"enabled": False},
            "aigc_defect": {"enabled": False},
            "product_fidelity": {"enabled": False},
        },
        "extractors": {"fps": 1, "max_frames": 16},
        "backends": {"mock": {"default_score": 0.75}},
        "fusion": {
            "strategy": "weighted_veto",
            "strict_veto_dims": [],
            "thresholds": {"A": 0.75, "B": 0.60, "C": 0.40},
            "veto_thresholds": {},
            "weights_general": {"technical_quality": 0.5, "hook_strength": 0.2, "cross_modal": 0.3},
        },
        "batch": {"mode": "resident", "chunk_size": 8},
        "output": {},
    }

    dm = DeviceManager("cpu")
    pipeline = Pipeline(config, dm)

    report = pipeline.run("/path/to/video.mp4", None, "general")
    print(f"Grade: {report.grade}, Score: {report.overall_score:.2f}")
    for dim, result in report.dimension_results.items():
        print(f"  {dim}: {result.score:.2f} ({result.status})")


if __name__ == "__main__":
    main()
