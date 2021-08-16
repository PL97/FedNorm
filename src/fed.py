import sys
import copy
import torch.optim as optim
import torch
import numpy as np


sys.path.extend(["../utils/", '../models/'])
from settings import parse_opts, write_log_head, write_log_body
from data_utils import prepare_data_fed, prepare_data, plot_fed, plot

from models import AlexNet, AlexNet_G, generate_models

def train(model, data_loader, optimizer, loss_fun, device):
    model.train()
    loss_all = 0
    total = 0
    correct = 0
    for data, target in data_loader:
        optimizer.zero_grad()

        data = data.to(device)
        target = target.to(device)
        output = model(data)
        loss = loss_fun(output, target)
        loss_all += loss.item()
        total += target.size(0)
        pred = output.data.max(1)[1]
        correct += pred.eq(target.view(-1)).sum().item()

        loss.backward()
        optimizer.step()

    return loss_all / len(data_loader), correct/total


def test(model, data_loader, loss_fun, device):
    model.eval()
    loss_all = 0
    total = 0
    correct = 0
    for data, target in data_loader:

        data = data.to(device)
        target = target.to(device)
        output = model(data)
        loss = loss_fun(output, target)
        loss_all += loss.item()
        total += target.size(0)
        pred = output.data.max(1)[1]
        correct += pred.eq(target.view(-1)).sum().item()

    return loss_all / len(data_loader), correct/total


def communication(args, server_model, models, client_weights):
    with torch.no_grad():
        # aggregate params
        if args.mode.lower() == 'fedbn':
            for key in server_model.state_dict().keys():
                if 'bn' not in key:
                    temp = torch.zeros_like(server_model.state_dict()[key], dtype=torch.float32)
                    for client_idx in range(args.client_num):
                        temp += client_weights[client_idx] * models[client_idx].state_dict()[key]
                    server_model.state_dict()[key].data.copy_(temp)
                    for client_idx in range(args.client_num):
                        models[client_idx].state_dict()[key].data.copy_(server_model.state_dict()[key])
        else:
            for key in server_model.state_dict().keys():
                # num_batches_tracked is a non trainable LongTensor and
                # num_batches_tracked are the same for all clients for the given datasets
                if 'num_batches_tracked' in key:
                    server_model.state_dict()[key].data.copy_(models[0].state_dict()[key])
                else:
                    temp = torch.zeros_like(server_model.state_dict()[key])
                    for client_idx in range(len(client_weights)):
                        temp += client_weights[client_idx] * models[client_idx].state_dict()[key]
                    server_model.state_dict()[key].data.copy_(temp)
                    for client_idx in range(len(client_weights)):
                        models[client_idx].state_dict()[key].data.copy_(server_model.state_dict()[key])
    return server_model, models


def fed_train(train_loaders, val_loaders, args, logfile):

	client_num = len(args.datasets)
	client_weights = [1/client_num for i in range(client_num)]

	server_model = generate_models(args).to(args.device)
	
	if args.use_gpu:
		server_model = torch.nn.DataParallel(server_model)
	client_models = [copy.deepcopy(server_model).to(args.device) for i in range(client_num)]


	# save by mean
	best_mean_acc = 0
	losses, accs = {}, {}
	losses['train'] = np.zeros((client_num, args.iters*args.wk_iters))
	losses['val'] = np.zeros((client_num, args.iters*args.wk_iters))
	accs['train'] = np.zeros((client_num, args.iters*args.wk_iters))
	accs['val'] = np.zeros((client_num, args.iters*args.wk_iters))

	for i in range(args.iters):
		optimizers = [optim.SGD(params=client_models[idx].parameters(), lr=args.lr) for idx in range(args.client_num)]
		for j in range(args.wk_iters):
			log_str_head = "============ Train epoch {}/{} ============".format(j + i * args.wk_iters, args.iters*args.wk_iters)
			print(log_str_head)
			# local update
			for client_idx, model in enumerate(client_models):
				train_loss, train_acc = train(model, train_loaders[client_idx], optimizers[client_idx], args.loss_func, args.device)

			# aggregation & save best model
			with torch.no_grad():
				log_str = ""
				server_model, models = communication(args, server_model, client_models, client_weights)

				# validation
				tmp_acc = np.zeros(client_num)
				for clients_index, model in enumerate(client_models):
					train_loss, train_acc = test(model, train_loaders[clients_index], args.loss_func, args.device)
					val_loss, val_acc = test(model, val_loaders[clients_index], args.loss_func, args.device)

					tmp_str = ' Site-{:<10s}| Train Loss: {:.4f} | Train Acc: {:.4f} | Val Loss: {:.4f} | Val Acc: {:.4f}'.format(args.datasets[clients_index] ,train_loss, train_acc, val_loss, val_acc)
					log_str += tmp_str + "\n"
					print(tmp_str)

					accs['train'][clients_index, j + i * args.wk_iters] = train_acc
					accs['val'][clients_index, j + i * args.wk_iters] = val_acc
					losses['train'][clients_index, j + i * args.wk_iters] = train_loss
					losses['val'][clients_index, j + i * args.wk_iters] = val_loss
					
				print("----"*15)

				# log to file
				write_log_body(logfile, log_str, j + i * args.wk_iters, args.iters*args.wk_iters)

				# save best model
				if best_mean_acc < np.mean(accs['val'][:, j + i * args.wk_iters]):
					best_mean_acc = np.mean(accs['val'][clients_index, j + i * args.wk_iters])
					tmp_str = "***update best model, save to {}***\n".format(args.model_path)
					logfile.write(tmp_str)
					print(tmp_str)

					saved_models = {}
					for tmp_idx, model in enumerate(client_models):
						saved_models['model_{}'.format(tmp_idx)] = model.state_dict()
					saved_models['server_model'] = server_model.state_dict()
					torch.save(saved_models, args.model_path)
	# plot
	plot_fed(args, losses, "losses")
	plot_fed(args, accs, "accuracy")


def pool_train(train_loader, val_loader, args, logfile):
	model = generate_models(args).to(args.device)
	if args.use_gpu:
		model = torch.nn.DataParallel(model)
	
	# save by mean
	losses, accs = {}, {}
	losses['train'] = np.zeros(args.iters*args.wk_iters)
	losses['val'] = np.zeros(args.iters*args.wk_iters)
	accs['train'] = np.zeros(args.iters*args.wk_iters)
	accs['val'] = np.zeros(args.iters*args.wk_iters)
	best_acc = 0


	optimizer = optim.SGD(params=model.parameters(), lr=args.lr)
	for i in range(args.iters):
		log_str_head = "============ Train epoch {}/{} ============".format(i, args.iters)
		print(log_str_head)
		train_loss, train_acc = train(model, train_loader, optimizer, args.loss_func, args.device)

		# aggregation & save best model
		with torch.no_grad():
			# validation
			log_str = ""
			train_loss, train_acc = test(model, train_loader, args.loss_func, args.device)
			val_loss, val_acc = test(model, val_loader, args.loss_func, args.device)
			tmp_str = ' Train Loss: {:.4f} | Train Acc: {:.4f} | Val Loss: {:.4f} | Val Acc: {:.4f}'.format(train_loss, train_acc, val_loss, val_acc)
			log_str = tmp_str + "\n"
			print(tmp_str)
			print("----"*15)
			# log to file
			write_log_body(logfile, log_str, i, args.iters)
			accs['train'][i] = train_acc
			accs['val'][i] = val_acc
			losses['train'][i] = train_loss
			losses['val'][i] = val_loss


			# save best model
			saved_models = {}
			if best_acc < val_acc:
				best_acc = val_acc
				tmp_str = "***update best model, save to {}***\n".format(args.model_path)
				logfile.write(tmp_str)
				print(tmp_str)
				saved_models['model'] = model.state_dict()
				torch.save(saved_models, args.model_path)

	# plot
	plot(args, losses, "losses")
	plot(args, accs, "accuracy")


def validate_fed(args, test_loader):
	client_num = len(args.datasets)
	server_model = generate_models(args).to(args.device)
	
	if args.use_gpu:
		server_model = torch.nn.DataParallel(server_model)
	client_models = [copy.deepcopy(server_model).to(args.device) for i in range(client_num)]

	saved = torch.load(args.model_path)
	server_model.load_state_dict(saved['server_model'])
	for i in range(client_num):
		client_models[i].load_state_dict(saved['model_{}'.format(i)])

	# test global model
	print("*********Result on global model*********")
	for clients_index, tl in enumerate(test_loader):
		test_loss, test_acc = test(server_model, tl, args.loss_func, args.device)
		tmp_str = ' Site-{:<10s}| Test Loss: {:.4f} | Test Acc: {:.4f}'.format(args.datasets[clients_index] ,test_loss, test_acc)
		print(tmp_str)
	print("*********Result on local model*********")
	for clients_index, tl in enumerate(test_loader):
		c_m = client_models[clients_index]
		test_loss, test_acc = test(c_m, tl, args.loss_func, args.device)
		tmp_str = ' Site-{:<10s}| Test Loss: {:.4f} | Test Acc: {:.4f}'.format(args.datasets[clients_index] ,test_loss, test_acc)
		print(tmp_str)

def validate(args, test_loader):
	model = generate_models(args).to(args.device)
	if args.use_gpu:
		model = torch.nn.DataParallel(model)
	model.load_state_dict(torch.load(args.model_path)['model'])
	for clients_index, tl in enumerate(test_loader):
		test_loss, test_acc = test(model, tl, args.loss_func, args.device)
		tmp_str = ' Site-{:<10s}| Test Loss: {:.4f} | Test Acc: {:.4f}'.format(args.datasets[clients_index] ,test_loss, test_acc)
		print(tmp_str)

if __name__ == "__main__":
	args = parse_opts()
	print(args)

	logfile = open(args.log_path, 'w')
	write_log_head(logfile, args)

	if args.test == True:
		if args.pool == False:
			test_loader = prepare_data_fed(args, train=False)
			validate_fed(args, test_loader)
		else:
			test_loader = prepare_data_fed(args, train=False)
			validate(args, test_loader)
	else:
		if args.pool == False:
			train_loaders, val_loaders = prepare_data_fed(args, debug=True)
			fed_train(train_loaders, val_loaders, args, logfile)

		else:
			train_loader, val_loader = prepare_data(args, debug=True)
			pool_train(train_loader, val_loader, args, logfile)