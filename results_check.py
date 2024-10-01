from pathlib import Path

def get_logs(p):
    p_logs = p / "log"
    logs = [x / "training.log" for x in p_logs.iterdir() if x.is_dir()]
    return logs

def get_best(p):
    logs = get_logs(p)
    BEST = 0
    for log in logs:
        with open(log, "r") as f:
            lines = f.readlines()
            # print last 4 lines
            target_line = lines[-3]
            
            # get the max_cor_B in target_line
            max_cor_B = float(target_line.split(" ")[6][:-1])
            if max_cor_B > BEST:
                BEST = max_cor_B
                BEST_LOG = log
    return BEST, BEST_LOG

if __name__ == "__main__":
    Parent = Path("./")
    for d in Parent.iterdir():
        if d.name.startswith("logs"):
            B, L = get_best(d)
            print(f"{d.name}: {B} in {L}")