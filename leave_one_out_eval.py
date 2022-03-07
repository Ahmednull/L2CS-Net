import os
import argparse



def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(
        description='gaze estimation using binned loss function.')
    parser.add_argument(
        '--evalpath', dest='evalpath', help='path for evaluating gaze test.',
        default="evaluation\L2CS-gaze360-_standard-10", type=str)
    parser.add_argument(
        '--respath', dest='respath', help='path for saving result.',
        default="evaluation\L2CS-gaze360-_standard-10", type=str)

if __name__ == '__main__':

    args = parse_args()
    evalpath =args.evalpath
    respath=args.respath
    if not os.path.exist(respath):
            os.makedirs(respath)
    with open(os.path.join(respath,"avg.log"), 'w') as outfile:    
        outfile.write("Average equal\n")

        min=10.0
        dirlist = os.listdir(evalpath)
        dirlist.sort()
        l=0.0
        for j in range(50):
            j=20
            avg=0.0
            h=j+3
            for i in dirlist:
                with open(evalpath+"/"+i+"/mpiigaze_binned.log") as myfile:
                    
                    x=list(myfile)[h]
                    str1 = "" 
                    
                    # traverse in the string  
                    for ele in x: 
                        str1 += ele  
                    split_string = str1.split("MAE:",1)[1] 
                    avg+=float(split_string)

            avg=avg/15.0
            if avg<min:
                min=avg
                l=j+1
            outfile.write("epoch"+str(j+1)+"= "+str(avg)+"\n")

        outfile.write("min angular error equal= "+str(min)+"at epoch= "+str(l)+"\n")
    print(min)
