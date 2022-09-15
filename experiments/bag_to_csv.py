#!/bin/python2

import rosbag, sys, csv, glob
import time
import string
import os #for file management make directory
import shutil #for file management, copy file


run = "zig_zag_fall"
listOfBagFiles = glob.glob("/home/spencer/Documents/DSL/CF_Data/sim2real_data_aug29_30/{}/*.bag".format(run))
numberOfFiles = len(listOfBagFiles)

count = 0
for bagFile in listOfBagFiles:
    count += 1
    print "reading file " + str(count) + " of  " + str(numberOfFiles) + ": " + bagFile
    #access bag
    bag = rosbag.Bag(bagFile)
    bagContents = bag.read_messages()
    bagName = bag.filename.split("/")[-1].rstrip(".bag")


    #create a new directory
    folder = "{}/data/{}".format(run, bagName)
    try:    #else already exists
        os.makedirs(folder)
    except:
        pass
    shutil.copyfile(bag.filename, folder + '/' + bagName)


    #get list of topics from the bag
    listOfTopics = []
    for topic, msg, t in bagContents:
        if topic not in listOfTopics:
            listOfTopics.append(topic)


    for topicName in listOfTopics:
        #Create a new CSV file for each topic
        filename = folder + '/' + string.replace(topicName, '/', '_slash_') + '.csv'
        with open(filename, 'w+') as csvfile:
            filewriter = csv.writer(csvfile, delimiter = ',')
            firstIteration = True    #allows header row
            for subtopic, msg, t in bag.read_messages(topicName):    # for each instant in time that has data for topicName
                #parse data from this instant, which is of the form of multiple lines of "Name: value\n"
                #    - put it in the form of a list of 2-element lists
                msgString = str(msg)
                msgList = string.split(msgString, '\n')
                instantaneousListOfData = []
                for nameValuePair in msgList:
                    splitPair = string.split(nameValuePair, ':')
                    for i in range(len(splitPair)):    #should be 0 to 1
                        splitPair[i] = string.strip(splitPair[i])
                    instantaneousListOfData.append(splitPair)
                #write the first row from the first element of each pair
                if firstIteration:    # header
                    headers = ["rosbagTimestamp"]    #first column header
                    for pair in instantaneousListOfData:
                        headers.append(pair[0])
                    filewriter.writerow(headers)
                    firstIteration = False
                # write the value from each pair to the file
                values = [str(t)]    #first column will have rosbag timestamp
                for pair in instantaneousListOfData:
                    if len(pair)>1:
                        values.append(pair[1])
                filewriter.writerow(values)
    bag.close()
print "Done reading all " + str(numberOfFiles) + " bag files."
