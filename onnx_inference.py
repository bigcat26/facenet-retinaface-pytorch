# import onnx
import numpy as np
import onnxruntime as ort

facenet = ort.InferenceSession("facenet.onnx", providers=['CPUExecutionProvider'])
facenet_sim = ort.InferenceSession("facenet_sim.onnx", providers=['CPUExecutionProvider'])

input = np.random.randn(1, 3, 160, 160).astype(np.float32)

outputs1 = facenet.run(None, {"input.1": input},)
outputs2 = facenet_sim.run(None, {"input.1": input},)

print(outputs1[0])
print(outputs2[0])

# model = onnx.load('facenet.onnx')

# onnx.checker.check_model(model)

# print(onnx.helper.printable_graph(model.graph))
