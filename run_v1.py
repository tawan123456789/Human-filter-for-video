import sys
import io
from pathlib import Path
import nbformat
from nbclient import NotebookClient

# Force UTF-8 output on Windows
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')
if sys.stderr.encoding != 'utf-8':
    sys.stderr.reconfigure(encoding='utf-8')

class StreamingClient(NotebookClient):
    def on_cell_executed(self, cell, cell_index, execute_reply):
        print(f"\n>>> Completed Cell {cell_index} ({cell.cell_type})", flush=True)
        for output in cell.get("outputs", []):
            if "text" in output:
                print(output["text"], end="", flush=True)
            elif "traceback" in output:
                print("\n".join(output["traceback"]), file=sys.stderr, flush=True)

NOTEBOOK_PATH = Path("notebooks/rvm_mobilenetv1_finetune.ipynb")

print("=" * 60)
print("Starting Execution: RVM-MobileNetV1 Fine-tuning")
print("Target Notebook   :", NOTEBOOK_PATH.resolve())
print("=" * 60, flush=True)

nb = nbformat.read(NOTEBOOK_PATH, as_version=4)
client = StreamingClient(
    nb,
    kernel_name="rvm_env",
    timeout=7200,
    resources={"metadata": {"path": str(Path("notebooks").resolve())}}
)

try:
    client.execute()
    nbformat.write(nb, NOTEBOOK_PATH)
    print("\n" + "=" * 60)
    print("FINISHED SUCCESSFULLY! All outputs saved to notebook.")
    print("=" * 60, flush=True)
except Exception as e:
    print(f"\nExecution encountered an error: {e}", file=sys.stderr, flush=True)
    # Save notebook even if there is an error to preserve outputs so far
    nbformat.write(nb, NOTEBOOK_PATH)
    sys.exit(1)
