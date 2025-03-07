import asyncio
from containers import Container
import logging

container = Container()

# ✅ Debugging: Check if YAML is loading correctly
try:
    container.config.from_yaml("config/config.yaml")
    print("✅ Config Loaded Successfully:")
    print(container.config())  # Should be a dictionary
    print("✅ Type of config:", type(container.config()))  # Should be <class 'dict'>
except Exception as e:
    print(f"❌ Error loading config: {e}")

async def main():
    try:
        data_processor = container.data_processor()
        print("✅ Data Processor Loaded Successfully")

        X, y = await data_processor.get_features_and_target()
        print("✅ Features and Target Loaded")
        print("🔹 X (features):", X)
        print("🔹 y (target):", y)

        triage_model = container.triage_model()
        print("✅ Triage Model Loaded Successfully")

        triage_model.train(X, y)
        evaluation_report = triage_model.evaluate(X, y)
        print("✅ Triage Model Evaluation:")
        print(evaluation_report)

        allocator = container.resource_allocator()
        print("✅ Resource Allocator Loaded Successfully")  # ✅ Added Success Message
        #print("🛠️ Available Methods:", [method for method in dir(allocator) if not method.startswith("__")])  # Clean Output

        await allocator.allocate_resources()

    except Exception as e:
        print(f"❌ An error occurred: {e}")

if __name__ == "__main__":
    asyncio.run(main())
