import pandas as pd

with open("x28.txt") as f:
    data = []
    content = f.readlines()[72:]
    for line in content:
        line  = line.strip()
        row = [float(i) for i in (line.split())]
        data.append(row)
df = pd.DataFrame(data)
print((df.head()))