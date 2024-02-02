lines = []

target_file = "metadata.jsonl"

with open(target_file) as fp:
    for line in fp:
        line = line.replace('"NaN"', 'NaN').replace('"null"', 'null')
        lines.append(line)

with open(target_file, "w") as fp:
    fp.writelines(lines)
