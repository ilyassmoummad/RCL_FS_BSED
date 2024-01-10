import torch
from torch.nn import functional as F
from losses import ProtoCLR
from train import adjust_learning_rate
from tqdm import tqdm

def get_distance(proto_pos, neg_proto, query_set_out):

    prototypes = torch.stack([proto_pos, neg_proto]).squeeze(1)
    dists = torch.cdist(query_set_out, prototypes)
    return dists


def finetune_proto(encoder, train_loader, transform,  args):

    print(f"Finetuning on {args.device}")
    
    loss_fn = ProtoCLR(tau=1.0)
    
    non_trainable_parameters = []
    
    # Uncomment below the layers to be frozen

    # for param in encoder.layer1.parameters():
    #     non_trainable_parameters.append(param)
    # for param in encoder.layer2.parameters():
    #     non_trainable_parameters.append(param)
    # for param in encoder.layer3.parameters():
    #     non_trainable_parameters.append(param)

    trainable_parameters = list( set(encoder.parameters()) - set(non_trainable_parameters) )

    optim = torch.optim.SGD(trainable_parameters, lr=args.ftlr, momentum=args.momentum, weight_decay=args.wd)
    
    num_epochs = args.ftepochs

    encoder = encoder.to(args.device)
    encoder.train()

    for epoch in range(1, num_epochs+1):
        tr_loss = 0.
        print("Epoch {}".format(epoch))

        adjust_learning_rate(optim, args.ftlr, epoch, num_epochs+1)
        train_iterator = iter(train_loader)

        for batch in tqdm(train_iterator):
            optim.zero_grad()
            
            x, y = batch
            x = x.to(args.device)
            y = y.to(args.device)

            x = transform(x); x2 = transform(x)

            _, x_out = encoder(x); _, x_out2 = encoder(x2)

            loss = loss_fn(x_out, x_out2, y)
            tr_loss += loss.item()

            loss.backward()
            optim.step()

                
        tr_loss = tr_loss/len(train_iterator)
        print('Average train loss: {}'.format(tr_loss))

    return encoder