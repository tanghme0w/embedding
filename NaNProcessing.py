lines = []

with open("metadata.jsonl") as fp:
    for line in fp:
        line = line.replace('"NaN"', 'NaN')
        lines.append(line)

with open("metadata.jsonl", "w") as fp:
    fp.writelines(lines)