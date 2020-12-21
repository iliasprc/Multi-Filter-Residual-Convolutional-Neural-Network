import torch
import numpy as np
from utils.utils import all_metrics, print_metrics
def train_gan(args, model_g,model_d, optimizer_g,optimizer_d, epoch, gpu, data_loader):

    print("EPOCH %d" % epoch)

    losses = []

    model_d.train_vae()
    model_g.train_vae()
    d_loss = 0
    g_loss = 0
    # loader
    data_iter = iter(data_loader)
    num_iter = len(data_loader)
    print(f'ITERATIONS {num_iter}')
    for i in range(num_iter):

        # if args.model.find("bert") != -1:
        #
        #     inputs_id, segments, masks, labels = next(data_iter)
        #     print('BERT RE')
        #
        #     inputs_id, segments, masks, labels = torch.LongTensor(inputs_id), torch.LongTensor(segments), \
        #                                          torch.LongTensor(masks), torch.FloatTensor(labels)
        #
        #     if gpu >= 0:
        #         inputs_id, segments, masks, labels = inputs_id.cuda(gpu), segments.cuda(gpu), \
        #                                              masks.cuda(gpu), labels.cuda(gpu)
        #
        #     output, loss = model_g(inputs_id, segments, masks, labels)
        # else:

        inputs_id, labels, text_inputs = next(data_iter)

        inputs_id, labels = torch.LongTensor(inputs_id), torch.FloatTensor(labels)

        if gpu >= 0:
            inputs_id, labels, text_inputs = inputs_id.cuda(gpu), labels.cuda(gpu), text_inputs.cuda(gpu)

        output, loss_g = model_g(inputs_id, labels, text_inputs)
        fake_out,fake_loss = model_d(output.detach(),False)
        real_out,real_loss = model_d(labels,True)

        _,loss_of_fake = model_d(output,True)
        loss_g+=0.1*loss_of_fake
        optimizer_g.zero_grad()
        loss_g.backward()
        optimizer_g.step()
        optimizer_d.zero_grad()
        (0.5*(fake_loss+real_loss)).backward()
        optimizer_d.step()
        losses.append(loss_g.item())
        d_loss +=(0.5*(fake_loss+real_loss)).item()
        g_loss+=loss_g.item()
        if i %100==0:
            print(f'loss d {d_loss/(i+1):.4f} loss g {g_loss/(i+1):.4f}')
    return losses

def test_gan(args, model_g, data_path, fold, gpu, dicts, data_loader):

    filename = data_path.replace('train', fold)
    print('file for evaluation: %s' % filename)
    num_labels = len(dicts['ind2c'])

    y, yhat, yhat_raw, hids, losses = [], [], [], [], []

    model_g.eval()

    # loader
    data_iter = iter(data_loader)
    num_iter = len(data_loader)
    for i in range(num_iter):
        with torch.no_grad():

            if args.model.find("bert") != -1:
                inputs_id, segments, masks, labels = next(data_iter)

                inputs_id, segments, masks, labels = torch.LongTensor(inputs_id), torch.LongTensor(segments), \
                                                     torch.LongTensor(masks), torch.FloatTensor(labels)

                if gpu >= 0:
                    inputs_id, segments, masks, labels = inputs_id.cuda(
                        gpu), segments.cuda(gpu), masks.cuda(gpu), labels.cuda(gpu)

                output, loss = model_g(inputs_id, segments, masks, labels)
            else:

                inputs_id, labels, text_inputs = next(data_iter)

                inputs_id, labels, = torch.LongTensor(inputs_id), torch.FloatTensor(labels)

                if gpu >= 0:
                    inputs_id, labels, text_inputs = inputs_id.cuda(gpu), labels.cuda(gpu), text_inputs.cuda(gpu)

                output, loss = model_g(inputs_id, labels, text_inputs)

            output = torch.sigmoid(output)
            output = output.data.cpu().numpy()

            losses.append(loss.item())
            target_data = labels.data.cpu().numpy()

            yhat_raw.append(output)
            output = np.round(output)
            y.append(target_data)
            yhat.append(output)

    y = np.concatenate(y, axis=0)
    yhat = np.concatenate(yhat, axis=0)
    yhat_raw = np.concatenate(yhat_raw, axis=0)

    k = 5 if num_labels == 50 else [8,15]
    metrics = all_metrics(yhat, y, k=k, yhat_raw=yhat_raw)
    print_metrics(metrics)
    metrics['loss_%s' % fold] = np.mean(losses)
    return metrics