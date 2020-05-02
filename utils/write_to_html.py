from pathlib import Path

OUT_FOLDER = 'html_out/'


def write_to_html(sentences, highlight_vals, filename, low_val=(255, 255, 255), high_val=(77, 145, 255),
                  out_dir=OUT_FOLDER):
    scaled_hl = [e * (1 / max(highlight_vals)) for e in highlight_vals]
    with open(Path(out_dir) / filename, 'w') as f:
        for sent, score in zip(sentences, scaled_hl):
            color_vals = [int(low*(1-score) + high*score) for low, high in zip(low_val, high_val)]
            f.write(f"<span style=\"background-color: rgb({color_vals[0]},{color_vals[1]},{color_vals[2]})\">"
                    f"{sent}</span>\n")


if __name__ == '__main__':
    write_to_html(["first_sentence", "second_setence", "third_sentence"], [0.6, 0.3, 0.1], "test.html")
