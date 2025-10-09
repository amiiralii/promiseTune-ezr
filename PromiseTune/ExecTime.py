from PromiseTune import *
import time

seeds = [0]
systems = ['LLVM', '7z', 'BDBC', 'exastencils', 'dconvert', 'deeparch',
            'PostgreSQL', 'javagc', 'storm', 'x264',
            'redis', 'HSQLDB']
# filename = sys.argv[1]
t1 = time.time()
for s in systems:
    filename = s + ".csv"
    t2 = time.time()
    run_main(seeds[0], filename)
    print(f"Time {filename}: {time.time()-t2:.2f}")
print(f"Time total: {time.time()-t1:.2f}")