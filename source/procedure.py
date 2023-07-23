import os
import numpy as np
import source.utils as utils

def training(agent, dataset, epochs, batch_size, normalize=True):

    utils.make_dir(path='result_tr', refresh=True)

    print("\nTraining to %d epochs (%d of minibatch size)" %(epochs, batch_size))

    iteration = 0
    for epoch in range(epochs):
        tmp_emb_s, tmp_emb_t, tmp_y = None, None, None
        while(True):
            minibatch = dataset.next_batch(batch_size=batch_size, ttv=0)

            if(minibatch['x1'].shape[0] == 0): break
            step_dict = agent.step(minibatch=minibatch, iteration=iteration, training=True)
            iteration += 1
            if(tmp_y is None):
                tmp_emb_s = step_dict['y_s']
                tmp_emb_t = step_dict['y_t']
                tmp_y = minibatch['y']
            else:
                tmp_emb_s = np.append(tmp_emb_s, step_dict['y_s'], axis=0)
                tmp_emb_t = np.append(tmp_emb_t, step_dict['y_t'], axis=0)
                tmp_y = np.append(tmp_y, minibatch['y'], axis=0)

            if(minibatch['terminate']): break

        agent.save_params(model='final_epoch')
        print("Epoch [%d / %d] (%d iteration)  Loss: %.5f" \
            %(epoch, epochs, iteration, step_dict['losses']['opt']))
        utils.plot_projection(tmp_emb_s, tmp_y, 1000, savepath=os.path.join('result_tr', 'epoch_%06d_s.pdf' %(epoch)))
        utils.plot_projection(tmp_emb_t, tmp_y, 1000, savepath=os.path.join('result_tr', 'epoch_%06d_t.pdf' %(epoch)))
    agent.save_params(model='final_epoch')

def test(agent, dataset, batch_size):

    list_model = utils.sorted_list(os.path.join(agent.path_ckpt, '*.pth'))
    for idx_model, path_model in enumerate(list_model):
        list_model[idx_model] = path_model.split('/')[-1]

    utils.make_dir(path='result_te', refresh=True)
    best_loss = 0
    for idx_model, name_model in enumerate(list_model):
        tmp_emb_s, tmp_emb_t, tmp_y, tmp_loss = None, None, None, None
        try: agent.load_params(model=name_model)
        except: print("Parameter loading was failed")
        else: print("Parameter loaded")

        print("\nTest... (w/ %s)" %(name_model))

        confusion_matrix = np.zeros((dataset.num_class, dataset.num_class), np.int32)
        while(True):

            minibatch = dataset.next_batch(batch_size=batch_size, ttv=1)
            if(minibatch['x1'].shape[0] == 0): break
            step_dict = agent.step(minibatch=minibatch, training=False)
            if(tmp_y is None):
                tmp_emb_s = step_dict['y_s']
                tmp_emb_t = step_dict['y_t']
                tmp_y = minibatch['y']
                tmp_loss = step_dict['losses']['opt_b']
            else:
                tmp_emb_s = np.append(tmp_emb_s, step_dict['y_s'], axis=0)
                tmp_emb_t = np.append(tmp_emb_t, step_dict['y_t'], axis=0)
                tmp_y = np.append(tmp_y, minibatch['y'], axis=0)
                tmp_loss = np.append(tmp_loss, step_dict['losses']['opt_b'], axis=0)

            if(minibatch['terminate']): break

        tmp_loss_avg = np.average(tmp_loss)
        print("Model: %s, Loss: %.5f" %(name_model, tmp_loss_avg))
        utils.plot_projection(tmp_emb_s, tmp_y, 1000, savepath=os.path.join('result_te', '%s_s.pdf' %(name_model)))
        utils.plot_projection(tmp_emb_t, tmp_y, 1000, savepath=os.path.join('result_te', '%s_t.pdf' %(name_model)))

        best_loss = max(best_loss, tmp_loss_avg)

    return best_loss, len(list_model)
