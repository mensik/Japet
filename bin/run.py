import subprocess

print "***********************************"

programs = ['testFeti', 'testSmale']
i = 0

for p in programs:
	print i,":", p, "\t",
	i = i + 1

i = int(raw_input("\n\nSelect program: "))
selectedProgram = programs[i]


x = ["Number of domain x", "-test_x", int, 2]
y = ["Number of domain y", "-test_y", int, 2]
h = ["Grid step", "-test_h", float, 2]
args = [x,y,h]
for arg in args:
	arg[4:4] = [arg[2](raw_input(arg[0] + ": (" + str(arg[3]) + ") "))]

carr = ["mpiexec", "-n", str(args[0][4] * args[1][4]), "./" + selectedProgram]
for arg in args:
	carr = carr + [arg[1], str(arg[4])]

subprocess.call(carr)
