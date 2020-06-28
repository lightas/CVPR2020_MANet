import os
f = open('inter_image.txt','w')
for seq in sorted(os.listdir('./')):
    if os.path.isdir(os.path.join('./',seq)):
        for inter in sorted(os.listdir(os.path.join('./',seq))):
            for img in os.listdir(os.path.join('./',seq,inter)):
                if img.split('_')[0]=='inter':
                    f.write(seq+' '+inter+' '+img+'\n')
f.close()

            
