library(rjags)

dat<-rbind(62, 60, 63, 59,
           63, 67, 71, 64, 65, 66,
           68, 66, 71, 67, 68, 68,
           56, 62, 60, 61, 63, 64, 63, 59
)

subID<-rbind(1, 1, 1, 1,
             2, 2, 2, 2, 2, 2,
             3, 3, 3, 3, 3, 3,
             4, 4, 4, 4, 4, 4, 4, 4
)

dat<-cbind(subID,dat)
colnames(dat)<-c("Subject","Value")
dat<-as.data.frame(dat)

#Format Data
Ndata = nrow(dat)
subj = as.integer(factor(dat$Subject, levels=unique(dat$Subject)))
Nsubj = length(unique(subj))
y = as.numeric(dat$Value)

# Can i put it to jags.model arguments?
#dataList = list(
#  Ndata = Ndata ,
#  Nsubj = Nsubj ,
#  subj = subj ,
#  y = y
#)


jags <- jags.model('normal_model.bug',
                   data = list('Ndata' = Ndata,
                               'subj' = subj,
                               'Nsubj' = Nsubj,
                               'y' = y),
                   n.chains = 4,
                   n.adapt = 100)

update(jags, 1000)

jags.samples(jags,
             c('theta', 'logtau', 'muG', 'logtauG'),
             1000)