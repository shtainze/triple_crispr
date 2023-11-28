signal.signal(signal.SIGINT, signal.SIG_DFL) # Ensure the calculation stops without error on Ctrl + C

# Open functions.py
with open("functions.py") as f:
    code = f.read()
    exec(code)
os.chdir(s_dir)
print('Last execution:', datetime.now())
