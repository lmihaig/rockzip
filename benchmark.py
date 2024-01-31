import json
import os
import pandas as pd
from timeit import default_timer as timer
import plotly.express as px

COMMANDS = [
    {"algo": "gzip", "args": "-c --keep", "options": ["-1", "-6", "-9"]},
    {"algo": "brotli", "args": "-c --keep", "options": ["-1", "-6", "-9"]},
    {"algo": "snzip", "args": "-c", "options": [""]},
    {"algo": "lz4", "args": "-c --keep", "options": ["-1", "-6", "-9"]},
    {"algo": "xz", "args": "-c --keep", "options": ["-1", "-6", "-9"]},
    {"algo": "bzip2", "args": "-c --keep", "options": ["-1", "-6", "-9"]},
    {"algo": "zstd", "args": "-c --keep", "options": ["-1", "-6", "-9"]},
    {"algo": "lzop", "args": "-c --keep", "options": ["-1", "-6", "-9"]},
]


CORPUS_PATH = "./corpus"
BENCHMAKR_PATH = "./benchmark"
RESULTS_FILE = f"{BENCHMAKR_PATH}/results.json"


def get_files(path):
    files = []
    for dirpath, dirnames, filenames in os.walk(path):
        files.extend(filenames)

    if files == []:
        raise Exception(f"No files found in corpus, check the path")
    return files


def run_benchmark(file, algo, args, options):
    results = {}
    base_output = f"{BENCHMAKR_PATH}/compressed/{algo}/"
    os.makedirs(base_output, exist_ok=True)
    for option in options:
        output = base_output + f"{file}_{algo}_{option.strip('-')}"
        real_file = f"{CORPUS_PATH}/{file}"

        print(f"Running {algo} {args} {option} {real_file} > {output}")
        start = timer()
        os.system(f"{algo} {args} {option} {real_file} > {output}")
        end = timer()

        results[f"option_{option.strip('-')}"] = {
            "time": end - start,
            "compressed_size": os.path.getsize(output),
        }

    return results


def load_results():
    if os.path.exists(RESULTS_FILE) and os.path.getsize(RESULTS_FILE) > 0:
        with open(RESULTS_FILE, "r") as file:
            try:
                return json.load(file)
            except json.JSONDecodeError:
                print(f"Error: The file {RESULTS_FILE} does not contain valid JSON.")
                exit(1)
    else:
        return {
            file_name: {
                "initial_size": os.path.getsize(os.path.join(CORPUS_PATH, file_name)),
                "results": {},
            }
            for file_name in get_files(CORPUS_PATH)
        }


def save_results(data):
    with open(RESULTS_FILE, "w") as file:
        json.dump(data, file)


def graph_results(data):
    for file, file_data in data.items():
        plot_data = []
        for algorithm, options in file_data["results"].items():
            for option, details in options.items():
                compression_ratio = (
                    details["compressed_size"] / file_data["initial_size"]
                ) * 100

                speed_mibs = file_data["initial_size"] / details["time"] / (2**20)

                plot_data.append(
                    {
                        "Algorithm": algorithm,
                        "Option": option,
                        "Compression Ratio (%)": compression_ratio,
                        "Speed (MiB/s)": speed_mibs,
                        "Time (s)": details["time"],
                    }
                )

        df = pd.DataFrame(plot_data)

        fig = px.scatter(
            df,
            x="Compression Ratio (%)",
            y="Speed (MiB/s)",
            color="Algorithm",
            symbol="Option",
            size="Time (s)",
            hover_data=["Algorithm", "Option"],
            title=f"Performance Comparison - {file}",
            labels={
                "Compression Ratio (%)": "Compression Ratio (%)",
                "Speed (MiB/s)": "Speed (MiB/s)",
            },
            template="plotly_dark",
        )

        fig.update_layout(
            legend_title_text="Algorithm",
            xaxis_title_text="Compression Ratio (%)",
            yaxis_title_text="Compression Speed (MiB/s)",
        )

        fig.update_traces(marker=dict(size=12))
        fig.write_image(f"{BENCHMAKR_PATH}/{file}.png")
        fig.show()


if __name__ == "__main__":
    data = load_results()
    # for command in COMMANDS:
    #     algo = command["algo"]
    #     args = command["args"]
    #     options = command["options"]

    #     data = load_results()

    #     for file_name in get_files(CORPUS_PATH):
    #         data[file_name]["results"][algo] = run_benchmark(
    #             file_name, algo, args, options
    #         )
    #         save_results(data)

    graph_results(data)
