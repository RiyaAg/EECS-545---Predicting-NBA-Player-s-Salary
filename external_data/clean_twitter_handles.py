

with open("twitter_handles.txt", "r") as f:
    
    lines = f.readlines()
    # print(lines)
    
    newlines = []
    
    for line in lines:
        splitline = line.split("\t")
        if splitline[0].lower() == "rk" or splitline[2] == "\n":
            continue
            
        
        newlines.append(",".join(splitline))
        
    # print(newlines)
    
with open("twitter_handles_clean.csv", "w") as clean:
    clean.writelines(newlines)