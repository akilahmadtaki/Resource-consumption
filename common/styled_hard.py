# styled_hard.py — a few HARD (7-object) puzzles, styled.
# Used to test whether the probe reacts to difficulty or merely to style.
# Same crisp-wording discipline to avoid reasoning loops.

styled_hard = [
    (0, "poetic", """Seven friends—fair Alice, Bob, and Carol too,
Dan, Eve, and Frank, and Grace to close the line—
Each keeps one beast, and each a different hue:
A cat, a dog, a fish, a bird so fine,
A rabbit, hamster, turtle—seven, true.
Now Alice—cat; and Bob—the dog we knew;
And Carol—bird; Dan—rabbit in the pen;
Eve—hamster; Frank holds not the fish. Say when:
Which of the seven, when the counting's done,
Is keeper of the fish—the final one?"""),

    (1, "song", """(Verse)
Henry, Iris, Jack, and Kate, and Leo too,
Mia and Noah, seven's the crew—
Red, blue, green, yellow, purple, orange, pink,
One color each, now stop and think.
Henry wears red, Iris wears blue,
Jack wears green, and Kate wears yellow too,
Leo's in purple, Mia ain't in orange, no—
(Chorus)
So who's wearin' orange? I gotta know!"""),

    (2, "archaic", """Hark! Seven minstrels there be—Olga, Paul, Quinn, Rosa, Sam, Tina, and Uma—and unto each belongeth one instrument alone, no two alike: to wit, a pianoforte, a violin, a drum, a flute, a guitar, a violoncello, and a harp. Know ye that Olga playeth the pianoforte, Paul the violin, Quinn the drum, Rosa the flute, and Sam the guitar; and that Tina playeth not the violoncello. Declare then: which among them playeth the violoncello?"""),
]

if __name__ == "__main__":
    print(f"{len(styled_hard)} styled-hard puzzles")
