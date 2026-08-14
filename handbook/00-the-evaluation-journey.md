# Why 90% Accuracy Doesn't Mean You Should Trust an AI Model

*The question about medical-AI metrics that finally made everything click.*

Throughout my career I've spent years building and evaluating predictive models and simulation systems. Whether the application was supply-chain optimization, computer vision, or CT imaging, evaluation metrics were never an afterthought — they were the backbone of deciding whether a model deserved to influence real-world decisions.

So when I began reading medical AI research, I expected the metrics to feel familiar. Accuracy, precision, recall — none of those surprised me. What surprised me was that every paper seemed to add another layer. AUROC. Calibration. Expected Calibration Error. Decision curves. External validation.

Eventually I stopped asking, *"What do these metrics mean?"* and started asking a different question:

> Why did researchers need to invent them in the first place?

## The number I wanted to believe

The first paper I read reported an accuracy above 90%, and I relaxed. Ninety percent *sounds* like a grade you'd be proud of — a solved problem, a model that mostly works.

Without noticing, I let that one number fill in blanks it had never earned. If it's 90% accurate, I assumed, it must also be trustworthy, reliable, confident for the right reasons, and safe enough for a doctor to lean on.

Accuracy promises none of that. It answers exactly one question:

> How often was the final prediction correct?

Nothing more. Every other quality I had attached to that number came from my imagination, not from the metric.

## A 90% model I wouldn't trust

It took me embarrassingly long to see the catch: you can post a headline accuracy that looks great and still have a model no clinician should touch. Let me show you two of them.

**The model that never finds the disease.** Picture a clinic screening 1,000 patients for a rare condition — only 10 of them actually have it. Now imagine the laziest model possible: it ignores every scan and labels *everyone* healthy.

Count its score. It is right 990 times out of 1,000 — **99% accuracy**. It also caught **zero** of the 10 sick patients. The one thing the model existed to do, it never did once. Suddenly 90-something percent doesn't sound reassuring at all — because when one class is rare, accuracy mostly measures how good you are at the easy answer.

**The model that's sure of everything.** Now two models that are both exactly 90% accurate. The first says *"I'm 90% confident"* and turns out right about 90% of the time — honest. The second announces *"I'm 99% confident"* on nearly every case, and is still right only 90% of the time.

Same accuracy. But the second model overstates its certainty every single time it speaks. Hand a doctor that 99% and they may skip a confirmatory test or start a treatment they'd otherwise have questioned. Accuracy simply cannot see the difference between the honest model and the reckless one.

Put those two together and you have your answer. A "90% accurate" model can be **blind** — high accuracy while it misses the disease entirely — or **dishonest** — accurate overall while wildly overstating its confidence. That is why the number, on its own, can't earn your trust. And it's why one number was never going to be enough.

## The metrics aren't competitors

For a while I read the growing list of metrics like a contest — accuracy versus F1, F1 versus AUROC, AUROC versus calibration — as if researchers just hadn't agreed on the best one yet. That isn't what's happening at all. Each metric exists to catch a specific failure the previous ones couldn't see — including the two you just met.

The *blindness* in the first example is exactly why **precision, recall, and F1** were invented: they ask how many sick patients you actually caught, not merely how often you were right. Judging a model at a single cutoff turned out not to be enough either, so **ROC curves and AUROC** ask how well it ranks patients across *every* threshold. And the *dishonesty* in the second example is why **calibration** exists — it checks whether "99% sure" really means ninety-nine-out-of-a-hundred.

Every new metric wasn't replacing the last one. It was covering a blind spot the last one couldn't see.

This isn't unique to medicine. Nobody ever accepted "the simulator is 87% accurate" and said "great, let's trust it." The room filled with other questions — what happens on Black Friday, what if demand doubles, does it systematically under-forecast? People weren't questioning the accuracy. They were questioning whether the model *deserved their trust*. Move to medicine and the vocabulary changes but the questions don't; only now the consequences are measured in patients rather than dollars.

It's how a clinician already thinks. Show a neurologist an AI that announces "Alzheimer's," and they almost never ask "what's the overall accuracy?" They ask: How certain is it? How often is it *confidently* wrong? Would it still work in my hospital? Those aren't versions of one question. They're different questions — and each metric exists to answer one of them.

## The evaluation journey

Once I saw it this way, the metrics stopped feeling like a pile of statistics and lined up into a sequence — a journey a model has to survive before anyone should trust it, where each stage asks a harder question than the last.

![The Evaluation Journey — five questions a model must survive; the first three are today's story.](figures/fig1_evaluation_journey.png)

The first three stages are the heart of the story.

**Stage 1 — Performance: can it make the right call?** Accuracy, precision, recall, F1 — all read off the confusion matrix, all judging the model *after* a threshold is chosen. This is where our lazy "everyone's healthy" model gets exposed: recall would have caught it instantly.

**Stage 2 — Ranking: can it separate sick from healthy?** Before you pick any threshold, you want to know whether the model can even sort patients by risk. That's AUROC — it summarizes performance across every possible cutoff at once, which is what you want when you're comparing models or ordering a worklist. *Its blind spot — the illusion of success:* an AUROC of 0.97 means the model ranks patients beautifully. It says nothing about whether the probabilities are honest.

**Stage 3 — Calibration: can we trust the probabilities?** This is where the overconfident model finally gets caught. When a model labels three patients "Alzheimer's" at 55%, 82%, and 99%, the diagnosis is identical but the care is not — you might monitor the first, refer the second, and open treatment discussions with the third. The probability itself changes medicine, so it had better mean what it says. Calibration measures exactly that: when the model says 90%, is it right about 90% of the time?

Two more stages wait beyond these — **Clinical Decision** (where do we actually draw the threshold?) and **Generalization** (will it survive the next hospital, with different scanners and different patients?). They matter enormously. But they're stories for later articles, not today's.

## Intelligence is not honesty

Here is the lesson that reframed everything for me: these qualities are *independent*.

![Intelligence is not honesty — ranking skill and probability-honesty are independent; a trustworthy model needs both.](figures/fig2_intelligence_not_honesty.png)

Our overconfident model ranks patients well and still lies about its confidence. A model that predicts the base rate for everyone is perfectly honest and completely useless. A model can look flawless at one threshold and fall apart at another. No single number captures all of that — which is precisely why one number was never going to be enough. A model that is *smart* is not automatically a model that is *honest*, and a model you can trust has to be both.

## How I read a paper now

These days I almost ignore the headline number at first. Instead I ask which questions the authors actually answered:

- Can it make good decisions, even when the disease is rare? *(Performance)*
- Can it separate high-risk patients from low-risk ones? *(Ranking)*
- Can its probabilities be trusted? *(Calibration)*

Only when those are answered *together* do I start to believe a model. The metrics were never competing for the title of "best." They collaborate — each tells one part of a story that's only complete when all the parts are told.

So no, 90% accuracy doesn't mean you should trust an AI model. It might be blind. It might be overconfident. The number alone can't tell you — and now you know which questions can.

*Next in this series: three deep dives into the metrics people misunderstand most — the F1 score, AUROC, and calibration — and why none of them can stand in for the others.*
