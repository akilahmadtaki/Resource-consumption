# styled_easy.py — the 6 easy puzzles, each rewritten in 3 styles.
# SAME logic, SAME answer as the plain version. Only the register changes.
# Wording kept crisp/unambiguous to avoid reasoning loops.

# Each entry: (puzzle_index, style_name, text)
styled_easy = [

    # ===== easy 0: Alice/Bob/Carol — pets (answer: Alice=fish) =====
    (0, "poetic", """Three friends—fair Alice, Bob, and Carol true—
Each keeps one pet, and each a different kind:
A cat, a dog, a fish, just three in view.
Bob keeps the dog, as clearly is defined.
And Alice, know, the cat she does not hold.
So tell me now, let the answer be told:
Which of the three, when all is said and done,
Is keeper of the fish—the final one?"""),
    (0, "song", """(Verse)
Alice, Bob, and Carol, three friends so fine,
Each with a pet, and each one's got a kind—
A cat, a dog, a fish, that's all, just three,
Bob's got the dog, plain as can be.
(Chorus)
Alice, she ain't holdin' the cat, oh no,
So who's got the fish? I really gotta know!"""),
    (0, "archaic", """Hark! Three companions there be—Alice, Bob, and Carol—and unto each belongeth one beast alone, no two alike: to wit, a cat, a hound, and a fish. Know ye that Bob possesseth the hound, and that Alice holdeth not the cat. Declare then: which among them keepeth the fish?"""),

    # ===== easy 1: Dan/Eve/Finn — fruit (answer: Dan=cherry) =====
    (1, "poetic", """Three children—Dan and Eve and Finn—we see,
Each picked one fruit, and each a different one:
An apple, banana, cherry, just those three.
Eve took banana, so that choice is done.
And Dan, be sure, no apple did he hold.
So tell me now, let the answer be told:
Which of the children, choosing from the tree,
Selected cherry? Name the one for me."""),
    (1, "song", """(Verse)
Dan and Eve and Finn, three kids in a row,
Each picked a fruit, here's what we know—
Apple, banana, cherry, just three,
Eve took banana, easy to see.
(Chorus)
Dan, he didn't pick the apple, no way,
So who's got the cherry? What do you say?"""),
    (1, "archaic", """Hark! Three young ones there be—Dan, Eve, and Finn—and unto each belongeth one fruit alone, no two alike: to wit, an apple, a banana, and a cherry. Know ye that Eve chose the banana, and that Dan chose not the apple. Declare then: which among them took the cherry?"""),

    # ===== easy 2: Grace/Henry/Ivy — subjects (answer: Grace=art) =====
    (2, "poetic", """Three students—Grace and Henry, Ivy too—
Each studies one, and each a different art:
'Tis math, 'tis history, 'tis art they pursue.
Now Henry, history—his chosen part.
And Grace, be sure, no math does she pursue.
So tell me now, as scholars often do:
Which of the three, with canvas or with chart,
Devotes her hours to the study of art?"""),
    (2, "song", """(Verse)
Grace and Henry, Ivy makes three,
Each studies something different, you see—
Math, history, art, just those three,
Henry takes history, easy and free.
(Chorus)
Grace, she isn't studying math, no no,
So who's doing art? I really wanna know!"""),
    (2, "archaic", """Hark! Three scholars there be—Grace, Henry, and Ivy—and unto each belongeth one study alone, no two alike: to wit, mathematics, history, and art. Know ye that Henry pursueth history, and that Grace pursueth not mathematics. Declare then: which among them studieth art?"""),

    # ===== easy 3: Jack/Kate/Liam — drinks (answer: Liam=tea) =====
    (3, "poetic", """Three people—Jack and Kate and Liam—stand,
Each with one drink, and each a different sort:
'Tis tea, 'tis coffee, juice—held in the hand.
Kate holds the coffee, so runs the report.
And Jack, be sure, no tea does he command.
So tell me now, and let it all be planned:
Which of the three, in cup of shortest sort,
Is holding tea? Now name him where they stand."""),
    (3, "song", """(Verse)
Jack and Kate and Liam, three in a line,
Each with a drink, and each one's just fine—
Tea, coffee, juice, that's all, just three,
Kate's got the coffee, plain as can be.
(Chorus)
Jack, he isn't holdin' the tea, oh no,
So who's got the tea? I really gotta know!"""),
    (3, "archaic", """Hark! Three folk there be—Jack, Kate, and Liam—and unto each belongeth one draught alone, no two alike: to wit, tea, coffee, and juice. Know ye that Kate holdeth the coffee, and that Jack holdeth not the tea. Declare then: which among them holdeth the tea?"""),

    # ===== easy 4: Mia/Noah/Olivia — colors (answer: Olivia=red) =====
    (4, "poetic", """Three artists—Mia, Noah, Olivia—paint,
Each with one hue, and each a different shade:
'Tis red, 'tis blue, 'tis green, without restraint.
Noah wields blue, and thus his choice is made.
And Mia, know, no red does she acquaint.
So tell me now, without a fault or feint:
Which of the three, when all the hues are laid,
Is she who paints in red? Let it be said."""),
    (4, "song", """(Verse)
Mia, Noah, Olivia, three who create,
Each with a color, and each one's great—
Red, blue, green, just those three,
Noah's using blue, easy to see.
(Chorus)
Mia, she isn't using red, no way,
So who's got the red? What do you say?"""),
    (4, "archaic", """Hark! Three painters there be—Mia, Noah, and Olivia—and unto each belongeth one hue alone, no two alike: to wit, red, blue, and green. Know ye that Noah useth blue, and that Mia useth not red. Declare then: which among them painteth in red?"""),

    # ===== easy 5: Paul/Quinn/Rosa — instruments (answer: Rosa=guitar) =====
    (5, "poetic", """Three players—Paul and Quinn and Rosa—play,
Each one instrument, each a different sound:
Guitar and piano, drums—in their array.
Quinn plays piano, so that much is found.
And Paul, be sure, guitar he does not sway.
So tell me now, ere music fades away:
Which of the three, when all the notes resound,
Is she who plays guitar? Let it be crowned."""),
    (5, "song", """(Verse)
Paul and Quinn and Rosa, three in the band,
Each with an instrument, each one's grand—
Guitar, piano, drums, just three,
Quinn's on piano, plain as can be.
(Chorus)
Paul, he isn't playin' guitar, oh no,
So who's on guitar? I really gotta know!"""),
    (5, "archaic", """Hark! Three minstrels there be—Paul, Quinn, and Rosa—and unto each belongeth one instrument alone, no two alike: to wit, a guitar, a pianoforte, and a drum. Know ye that Quinn playeth the pianoforte, and that Paul playeth not the guitar. Declare then: which among them playeth the guitar?"""),
]

if __name__ == "__main__":
    print(f"{len(styled_easy)} styled-easy puzzles")
    # sanity: 6 puzzles x 3 styles = 18, and count per style
    from collections import Counter
    print("per style:", dict(Counter(s for _, s, _ in styled_easy)))
