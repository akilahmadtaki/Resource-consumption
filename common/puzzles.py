# puzzles.py — plain puzzle dataset, labeled by difficulty (object count)
# Every puzzle is structurally identical within its difficulty:
#   assign most pets directly, leave the last one to a single negative clue.
# All are verified solvable with a unique answer (answer noted in comments).

easy_puzzles = [   # 3 objects
    "Three friends—Alice, Bob, and Carol—each have a different pet: a cat, a dog, and a fish. Bob has the dog. Alice does not have the cat. Who has the fish?",          # Alice
    "Three children—Dan, Eve, and Finn—each chose a different fruit: an apple, a banana, and a cherry. Eve chose the banana. Dan did not choose the apple. Who chose the cherry?",  # Dan
    "Three students—Grace, Henry, and Ivy—each study a different subject: math, history, and art. Henry studies history. Grace does not study math. Who studies art?",   # Grace
    "Three people—Jack, Kate, and Liam—each ordered a different drink: tea, coffee, and juice. Kate ordered the coffee. Jack did not order the tea. Who ordered the tea?", # Liam
    "Three artists—Mia, Noah, and Olivia—each used a different color: red, blue, and green. Noah used blue. Mia did not use red. Who used red?",                          # Olivia
    "Three musicians—Paul, Quinn, and Rosa—each play a different instrument: guitar, piano, and drums. Quinn plays piano. Paul does not play guitar. Who plays guitar?",   # Rosa
]

medium_puzzles = [   # 5 objects
    "Five friends—Alice, Bob, Carol, Dan, and Eve—each have a different pet: a cat, a dog, a fish, a bird, and a rabbit. Alice has the cat. Bob has the dog. Carol has the bird. Dan does not have the fish. Who has the fish?",                    # Eve
    "Five people—Frank, Grace, Henry, Iris, and Jack—each wear a different color: red, blue, green, yellow, and purple. Frank wears red. Grace wears blue. Henry wears green. Iris does not wear yellow. Who wears yellow?",                        # Jack
    "Five musicians—Kate, Leo, Mia, Noah, and Olga—each play a different instrument: piano, violin, drums, flute, and guitar. Kate plays piano. Leo plays violin. Mia plays drums. Noah does not play flute. Who plays flute?",                    # Olga
    "Five students—Paul, Quinn, Rosa, Sam, and Tina—each study a different subject: math, history, art, biology, and chemistry. Paul studies math. Quinn studies history. Rosa studies art. Sam does not study biology. Who studies biology?",     # Tina
    "Five children—Uma, Victor, Wendy, Xavier, and Yara—each picked a different fruit: an apple, a banana, a cherry, a date, and an elderberry. Uma picked the apple. Victor picked the banana. Wendy picked the cherry. Xavier picked the elderberry. Who picked the date?",  # Yara
    "Five athletes—Adam, Beth, Cody, Dora, and Evan—each play a different sport: soccer, tennis, golf, rugby, and hockey. Adam plays soccer. Beth plays tennis. Cody plays golf. Dora does not play rugby. Who plays rugby?",                      # Evan
]

hard_puzzles = [   # 7 objects
    "Seven friends—Alice, Bob, Carol, Dan, Eve, Frank, and Grace—each have a different pet: a cat, a dog, a fish, a bird, a rabbit, a hamster, and a turtle. Alice has the cat. Bob has the dog. Carol has the bird. Dan has the rabbit. Eve has the hamster. Frank does not have the fish. Who has the fish?",                    # Grace
    "Seven people—Henry, Iris, Jack, Kate, Leo, Mia, and Noah—each wear a different color: red, blue, green, yellow, purple, orange, and pink. Henry wears red. Iris wears blue. Jack wears green. Kate wears yellow. Leo wears purple. Mia does not wear orange. Who wears orange?",   # Noah
    "Seven musicians—Olga, Paul, Quinn, Rosa, Sam, Tina, and Uma—each play a different instrument: piano, violin, drums, flute, guitar, cello, and harp. Olga plays piano. Paul plays violin. Quinn plays drums. Rosa plays flute. Sam plays guitar. Tina does not play cello. Who plays cello?",   # Uma
    "Seven students—Victor, Wendy, Xavier, Yara, Zack, Amy, and Ben—each study a different subject: math, history, art, biology, chemistry, physics, and geography. Victor studies math. Wendy studies history. Xavier studies art. Yara studies biology. Zack studies chemistry. Amy does not study physics. Who studies physics?",   # Ben
    "Seven children—Cara, Drew, Ella, Finn, Gus, Hana, and Ian—each picked a different fruit: an apple, a banana, a cherry, a date, an elderberry, a fig, and a grape. Cara picked the apple. Drew picked the banana. Ella picked the cherry. Finn picked the date. Gus picked the elderberry. Hana did not pick the fig. Who picked the fig?",   # Ian
    "Seven athletes—Jill, Kyle, Lena, Max, Nora, Omar, and Pia—each play a different sport: soccer, tennis, golf, rugby, hockey, cricket, and baseball. Jill plays soccer. Kyle plays tennis. Lena plays golf. Max plays rugby. Nora plays hockey. Omar does not play cricket. Who plays cricket?",   # Pia
]
