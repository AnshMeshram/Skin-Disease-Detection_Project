Train base model
python main.py --stage train --config config.yaml --model efficientnet_b3

(Optional) Resume training later
python main.py --stage train --config config.yaml --model efficientnet_b3 --resume

Evaluate trained model
python main.py --stage evaluate --config config.yaml --model efficientnet_b3

Generate Grad-CAM validation visuals
python main.py --stage gradcam --config config.yaml --model efficientnet_b3 --fold 0 --n_per_class 3

Train other two models for ensemble
python main.py --stage train --config config.yaml --model inception_v3
python main.py --stage train --config config.yaml --model convnext_tiny

Run ensemble optimization + evaluation
python main.py --stage ensemble --config config.yaml

Single-image JSON prediction
python main.py --stage predict --config config.yaml --image path/to/img.jpg --output pred.json

Live webcam inference
python main.py --stage live --config config.yaml --model efficientnet_b3
or
python main.py --stage live --config config.yaml --use_ensemble

Start REST API
uvicorn src.api:app --host 0.0.0.0 --port 8080 --reload

Test API
curl -X POST -F "file=@test/lesion.jpg" http://localhost:8080/predict

Important:

Steps 7 to 10 require trained checkpoints.
Right now, your environment is validated; next actionable step is Step 1 (training).
GPT-5.3-Codex • 0.9x
