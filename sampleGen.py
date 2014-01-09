from random import uniform

def sampleGen(filename):
    wf = open(filename,"w")
    buf = ["chr1 " + str(i) + " " + str(uniform(3,5)) + " " + str(uniform(0,2)) + "\n" for i in range(100,200)]
    wf.writelines(buf)
    wf.close()

if __name__ == "__main__":
    sampleGen("sample.txt")
