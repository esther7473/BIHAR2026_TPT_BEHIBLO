from src.inference.inference import run_inference

forecast_index, forecast = run_inference()

print(forecast_index[:5])
print(forecast[:5])



