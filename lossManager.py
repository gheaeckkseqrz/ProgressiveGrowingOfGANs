import torch
import visdom
#import matplotlib as mpl
#mpl.use('Agg') # -- To work without a X server
#import matplotlib.pyplot as plt
#import mail
import sys
import collections

class LossManager():

    def __init__(self, intervals = [100, 500, 1000, 10000, 100000], backlog = 100000, displayEvery = 100, win = "Losses", env = None):
        assert(intervals[-1] <= backlog)
        self.intervals = intervals
        self.offset = 0
        self.total = 0
        self.data = torch.zeros(len(self.intervals) + 1, backlog)
        self.vis = visdom.Visdom()
        self.win = win
        self.env = env
        # self.mailClient = mail.EmailLogger()
        self.displayEvery = displayEvery
        self.emailEvery = -1
        self.reset()

    def reset(self):
        self.data.fill_(0)
        self.total = 0
        self.offset = 0
        self.buffers = []
        self.sums = []
        for i in range(len(self.intervals)):
            self.buffers.append(collections.deque())
            self.sums.append(0)

    def registerLoss(self, l):
        self.data[0][self.offset] = l
        self.offset = (self.offset + 1) % self.data.size(1)
        self.total +=1
        for i in range(len(self.intervals)):
            self.buffers[i].append(l)
            self.sums[i] += l
            if len(self.buffers[i]) > self.intervals[i]:
                self.sums[i] -= self.buffers[i].popleft()
            self.data[i+1][self.offset-1] = self.sums[i] / len(self.buffers[i])
        if self.total % self.displayEvery == 0 and self.displayEvery > 0:
            self.display()

    def unrollTensor(self):
        if self.total == 0:
            return self.data[:,0:1]
        if self.offset >= self.data.size(1) or self.offset == 0:
            return self.data
        else:
            before = self.data[:,0:self.offset]
            if self.total < self.data.size(1):
                return before
            else:
                after = self.data[:,self.offset:]
                return torch.cat([after, before], 1)

    def hasConverged(self):
        if self.total < self.data.size(1):
            return False ## Let's assume loss didn't converge in so little time
        return self.data[len(self.intervals)][self.offset - 1] > self.data[len(self.intervals)][(self.offset + 1) % self.data.size(1)]

    # def sendEmail(self):
    #     attachement = self.display()
    #     subject = self.win
    #     body = " [Iteration " + str(self.total) + "] - Current loss is " + str(self.data[len(self.intervals)][self.offset-1])
    #     if self.hasConverged():
    #         body += " - TRAINING DONE"
    #     self.mailClient.send(subject, body, [attachement])

    def display(self, ax = None):
        try:
            if self.total < 2:
                return
            legends = ["Data"]
            for i in self.intervals:
                legends.append("Moving Average " + str(i))
            self.vis.line(Y=self.unrollTensor().t(),
                          X=torch.arange(max(0,self.total - self.data.size(1)), self.total),
                          opts=dict(legend=legends, title=self.win, width=1700, height=600),
                          win=self.win,
                          env=self.env
            )
        except:
            print("Visdom code threw exception")
            pass

        # try:
        #     plotName = "LOSSES_GRAPH/" + self.win.replace(" ", "_") + "_loss_" + str(self.total) + ".png"
        #     if ax == None:
        #         fig, ax = plt.subplots( nrows=1, ncols=1, figsize=(20,10))  # create figure & 1 axis
        #     ax.set_title(self.win)
        #     X = range(max(0,self.total - self.data.size(1)) + 1, self.total + 1)
        #     for i in range(len(self.intervals)):
        #         Y = self.unrollTensor()[i+1].numpy()
        #         ax.plot(X, Y, label='Moving Average '+str(self.intervals[i]))
        #     leg = ax.legend(loc='best', ncol=1)
        #     leg.get_frame().set_alpha(1)
        #     fig.savefig(plotName)   # save the figure to file
        #     plt.close('all')    # close the figure
        #     return plotName
        # except Exception as e:
        #     print >> sys.stderr, "Matplotlib code threw exception ["+str(e)+"]"
        #     pass




