import pyedflib
import numpy as np
import os

# f = pyedflib.EdfReader("C:\\Users\\Omar\\Desktop\\sleep-edf-database-expanded-1.0.0\\sleep-cassette\\SC4001E0-PSG.edf")
# f = pyedflib.EdfReader("SC4001EC-Hypnogram.edf")
# print(f.getSignalLabels())
# print(f.getNSamples())
# print("file duration: %i seconds" % f.file_duration)
# print("startdate: %i-%i-%i" % (f.getStartdatetime().day, f.getStartdatetime().month, f.getStartdatetime().year))
# print("number of annotations in the file: %i" % f.annotations_in_file)

# codeName = "SC4001E0"
# psgName = "SC4001E0-PSG"
# hypnogramName = "SC4001EC-Hypnogram"
for filename in os.listdir('Dataset_EEG/data_raw/sleep/sleep-cassette'):
    if filename.endswith("-PSG.edf"):
        codeName = filename[0:8]
        psgName = codeName + '-PSG'
        continue
    print(codeName)
    hypnogramName = filename[0:8] + '-Hypnogram'
    PSG = pyedflib.EdfReader('./sleep/sleep-cassette/' + psgName + '.edf')
    Hypnogram = pyedflib.EdfReader('./sleep/sleep-cassette/' + hypnogramName + '.edf')
    n = PSG.signals_in_file
    Fpz_Cz = PSG.readSignal(0)
    Pz_Oz = PSG.readSignal(1)
    epoch = 30*100
    os.makedirs('data_cooked/%s' % codeName)

    sW = -1; s1 = -1; s2 = -1; s3 = -1; s4 = -1; sR = -1; counter = -1
    annotations = Hypnogram.readAnnotations()
    for annotation in np.arange(Hypnogram.annotations_in_file):
        print("annotation: onset is %f    duration is %s    description is %s" % (annotations[0][annotation],annotations[1][annotation],annotations[2][annotation]))
        if annotations[2][annotation] == "Sleep stage 1":
            s1 += 1
            counter = s1
        elif annotations[2][annotation] == "Sleep stage 2":
            s2 += 1
            counter = s2
        elif annotations[2][annotation] == "Sleep stage 3":
            s3 += 1
            counter = s3
        elif annotations[2][annotation] == "Sleep stage 4":
            s4 += 1
            counter = s4
        elif annotations[2][annotation] == "Sleep stage W":
            sW += 1
            counter = sW
        elif annotations[2][annotation] == "Sleep stage R":
            sR += 1
            counter = sR
        elif annotations[2][annotation] == "Sleep stage ?":
            continue
        csvFile = open('./data_cooked/%s/%s.csv' % (codeName, codeName+"-"+annotations[2][annotation]+"-"+str(counter)), 'w')
        csvFile.write(PSG.getSignalLabels()[0] + "," + PSG.getSignalLabels()[1] + "\n")
        for signal in np.arange(int(annotations[0][annotation])*100, int(annotations[0][annotation])*100 + int(annotations[1][annotation]*100)):
            csvFile.write(str(Fpz_Cz[int(signal)]) + "," + str(Pz_Oz[int(signal)]) + "\n")
        csvFile.close()

# print("\nread %i samples\n" % epoch)
# result = ""
# for i in np.arange(epoch):
#     result += ("%.1f, " % Pz_Oz[i])
# print(result)


# sigbufs = np.zeros((2, f.getNSamples()[0]))
# for i in np.arange(n):
#     sigbufs[i, :] = f.readSignal(i)
