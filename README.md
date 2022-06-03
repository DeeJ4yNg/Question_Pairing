########################### Question_Pairing ###########################

This is a code block for a Sentences Matching Task in an NLP project.

It filters the top 5 similarity questions in data set and completes the first phase filter work for the task.

The code will:

1. Read data from a cleaned data set(like below).

                      Questions                                             Answers
         How can I do if I forgot my password?                    Contact IT dept to reset.
         I forgot my password.
         I can't logon my computer.
         .....
         
2. Tokenize words, output sentence vector (Pretrained model).
3. Tokenize input question, output sentence vector. 
4. Compute Cosine Similarity between input question and questions in data set.
5. Output the max one.
6. SVD the sentence vectors and plot on a picture.

#########################################################################
