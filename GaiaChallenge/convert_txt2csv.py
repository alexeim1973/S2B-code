fin = open("GaiaChallenge/modelR1GaiaChallenge", "rt")
fout = open("GaiaChallenge/modelR1GaiaChallenge.csv2", "wt")

for line in fin:
	fout.write(','.join(line.split())+'\n')
	
fin.close()
fout.close()