There are 2 versions of the CNN: 
	1 which uses the Cross-Entropy loss function
	1 which uses the Coefficient of determination loss function

----TO RUN SUCCESSFULLY----

-To run this you need to create 2 empty folders in the working directory for them to store the data into. Name them 'coef' & 'ent' (otherwise you will get an error)
-Each CNN code has a "SETTINGS" section near the top (you can do a word-search for "Settings" to find it).  This allows you to change the epochs, batchsize, images being utilized, and validation/test size




----YOUR OUTPUT----
- The 'coef' and 'ent' files will now have the results of the running the codes. The latter gives the results of the  'Cross-entropy loss' and the former gives the result for the 'Coefficient loss'
- Each folder will have the following

1.  Train and test accuracy per each epoch for Adam
2.  Report listing the final precision, recall, and fscore of Adam
3.  Train and test accuracy per each epoch for SGD
4.  Report listing the final precision, recall, and fscore of SGD
5.  Train and test accuracy per each epoch for Adagrad
6.  Report listing the final precision, recall, and fscore of Adagrad
7.  Train and test accuracy per each epoch for RMSprop
8.  Report listing the final precision, recall, and fscore of RMSpropr




-If you want plots, there is commented code at the bottom of each file that can read off the relevant text files and produce them.