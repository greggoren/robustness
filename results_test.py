bins = {}
jumps = 0.1
start = 0
end = 0.1
for i in range(10):
    bins[(start, end)] = 0
    start = round(end, 3)
    end += jumps
    end = round(end, 3)

print(bins)
