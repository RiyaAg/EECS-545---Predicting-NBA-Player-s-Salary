

with open("igfollowers.csv", "r") as f:
    lines = f.readlines()
    newlines = [line.replace(",", "").replace("\t", ",") for line in lines]
    print(newlines)
    # print(lines)
    
    
with open("igfollowersclean.csv", "w") as clean:
    clean.writelines(newlines)