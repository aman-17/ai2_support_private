import sys

def split_revisions(revisions_file: str, n: int):
    with open(revisions_file) as f:
        revisions = f.readlines()

    total = len(revisions)
    subset = total // n

    for i in range(n):
        start = i * subset
        end = start + subset
        if i == n - 1:
            end = None
        print(start, end)
        subset_file = revisions_file.replace(".txt", f"-{i+1}.txt")
        with open(subset_file, "w") as f:
            for revision in revisions[start:end]:
                f.write(revision)

if __name__ == "__main__":
    split_revisions(sys.argv[1], int(sys.argv[2]))

