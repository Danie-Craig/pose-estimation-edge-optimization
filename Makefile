.PHONY: verify inference benchmark robustness

verify:
	python scripts/verify_setup.py

inference:
	python scripts/run_inference.py --source data/test_video1.mp4 --save-video --output-dir results/inference

benchmark:
	python scripts/run_benchmark.py --model lightweight --runs 200

robustness:
	python scripts/evaluate_robustness_filtered.py --max-images 2693 --output-dir results/robustness_coco