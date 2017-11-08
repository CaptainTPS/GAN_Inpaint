import shutil

import train_random
import random_test

def main():
    path = '/home/cad/PycharmProjects/ContextEncoder/ganTest/'

    for i in range(1, 120):
        maskpath = path + 'gan_'+str(i)+'_mask.png'
        imgpath = path +  'gan_'+str(i)+'_img.png'

        #copy to test folder
        testfolder = "/home/cad/PycharmProjects/ContextEncoder/ganTest/loadData/data/"
        for x in range(70):
            toPath = testfolder + str(x) + ".png"
            shutil.copy(imgpath, toPath)

        #run training
        train_random.main(maskpath)
        #run test
        random_test.main(maskpath)
        #save result
        oripath = "testoutput/gen.png"
        shutil.copy(oripath, path + "output/" + str(i) + ".png")
        print("dealing to " + str(i))
        # break

if __name__ == "__main__":
    main()