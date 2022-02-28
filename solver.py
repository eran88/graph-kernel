import numpy as np
import os
import time
import datetime
from pathlib import Path
from collections import deque
import torch
import torch.nn.functional as F
from models import Learned_laplacian_model,Generator, Discriminator,Learned_alpha_model
from data.sparse_molecular_dataset import SparseMolecularDataset
from utils import *

class Solver(object):

    def __init__(self, config):
        """Initialize configurations."""
        self.shapes=config.shapes
        self.name=config.name
        # Data loader.
        self.data = SparseMolecularDataset()
        self.data.load(config.mol_data_dir)

        # Model configurations.
        self.z_dim = config.z_dim
        self.m_dim = self.data.atom_num_types
        self.b_dim = self.data.bond_num_types
        self.g_conv_dim = config.g_conv_dim
        self.d_conv_dim = config.d_conv_dim
        self.g_repeat_num = config.g_repeat_num
        self.d_repeat_num = config.d_repeat_num
        self.post_method = config.post_method
        self.random_graph_increase=config.random_graph_increase
        self.change_alpha=config.change_alpha
        self.metric = 'qed,validity,sas'
        self.learned_alpha=config.learned_alpha
        # Training configurations.
        self.batch_size = config.batch_size
        self.num_iters = config.num_iters
        self.num_iters_decay = config.num_iters_decay
        self.g_lr = config.g_lr
        self.d_lr = config.d_lr
        self.dropout = config.dropout
        self.n_critic = config.n_critic
        self.beta1 = config.beta1
        self.beta2 = config.beta2
        self.alpha = config.alpha
        self.laplacian_alpha = config.laplacian_alpha
        self.shapes_alpha = config.shapes_alpha
        self.learned_shapes=config.learned_shapes
        self.learned_laplacian=config.learned_laplacian
        self.alphas=[config.alpha]
        min_alpha=config.alpha
        max_alpha=config.alpha
        if config.alpha_range>0:
            self.multi_alpha=True
        else:
            self.multi_alpha=False
        for i in range(config.alpha_range):
            min_alpha=min_alpha/config.alpha_mul
            max_alpha = max_alpha * config.alpha_mul
            self.alphas.append(min_alpha)
            self.alphas.append(max_alpha)
        self.min_loss=config.min_loss
        self.laplacian=config.laplacian
        if config.num_learned_kernel>0:
            self.learned_kernel=True
        else:
            self.learned_kernel = False
        self.num_disc=config.num_learned_kernel+config.num_replacing_kernel+config.num_constant_kernel
        if self.num_disc<=0:
            raise Exception('You must use at least 1 GCN kernel!')

        self.num_learned_kernel=config.num_learned_kernel
        self.num_replacing_kernel=config.num_replacing_kernel
        self.num_constant_kernel=config.num_constant_kernel

        # Miscellaneous.
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.good_mols=deque(maxlen=50)
        self.bad_mols=deque(maxlen=50)
        self.valid50=deque(maxlen=50)
        self.unique_valid50=deque(maxlen=50)

        # Directories.
        main_dir="results/"+config.name
        self.main_dir=main_dir
        self.log_dir = main_dir+"/logs"
        self.model_save_dir =  main_dir+"/models"
        self.samples =  main_dir+"/fake_samples"
        self.real_samples =  main_dir+"/real_samples"
        self.samples_iter = config.samples_iter
        self.print_file= main_dir+"/logs.txt"
        print_file_parent = Path(self.print_file).parent
        Path("results").mkdir(exist_ok=True)
        Path(print_file_parent).mkdir(exist_ok=True)
        open(self.print_file, 'w').close()
        Path(main_dir).mkdir(exist_ok=True)
        Path(self.samples).mkdir(exist_ok=True)
        Path(self.real_samples).mkdir(exist_ok=True)
        Path(self.log_dir).mkdir(exist_ok=True)
        Path(self.model_save_dir).mkdir(exist_ok=True)

        # Step size.
        self.log_step = config.log_step
        self.model_save_step = config.model_save_step
        self.lr_update_step = config.lr_update_step

        # Build the model and tensorboard.
        self.build_model()
    def build_discriminator(self):
        self.D=[]
        params=[]
        for i in range (self.num_disc):
            self.D.append(Discriminator(self.d_conv_dim, self.m_dim, self.b_dim, self.dropout,self.device,self.random_graph_increase))
        for i in range(self.num_learned_kernel):
            params=params+list(self.D[i].parameters())
        if self.learned_alpha:
            self.Learned_alpha_model=Learned_alpha_model(self.alpha)
            params = params + list(self.Learned_alpha_model.parameters())
            self.Learned_alpha_model.to(self.device)
        else:
            self.Learned_alpha_model=False
        if self.learned_laplacian:
            self.Learned_laplacian_model=Learned_laplacian_model(9)
            params = params + list(self.Learned_laplacian_model.parameters())
            self.Learned_laplacian_model.to(self.device)
        else:
            self.Learned_laplacian_model=False
        if self.learned_shapes:
            self.Learned_shapes_model=Learned_laplacian_model(10)
            params = params + list(self.Learned_shapes_model.parameters())
            self.Learned_shapes_model.to(self.device)
        else:
            self.Learned_shapes_model=False
        if self.learned_kernel or self.learned_alpha:
          self.d_optimizer = torch.optim.Adam(params, self.d_lr, [self.beta1, self.beta2])
        #self.print_network(self.D, 'D')
        for i in range (self.num_disc):
            self.D[i].to(self.device)
    def Dtrain(self):
        for i in range(self.num_disc):
            self.D[i].train()
    def build_model(self):
        """Create a generator and a discriminator."""
        self.G = Generator(self.g_conv_dim, self.z_dim,
                           self.data.vertexes,
                           self.data.bond_num_types,
                           self.data.atom_num_types,
                           self.dropout)


        self.g_optimizer = torch.optim.Adam(list(self.G.parameters()),self.g_lr, [self.beta1, self.beta2])
        self.print_network(self.G, 'G')
        self.G.to(self.device)
        self.build_discriminator()

    def print_network(self, model, name):
        """Print out the network information."""
        num_params = 0
        for p in model.parameters():
            num_params += p.numel()
        print(model)
        print(name)
        print("The number of parameters: {}".format(num_params))






    def update_lr(self, g_lr, d_lr):
        """Decay learning rates of the generator and discriminator."""
        for param_group in self.g_optimizer.param_groups:
            param_group['lr'] = g_lr
        if self.learned_kernel:
            for param_group in self.d_optimizer.param_groups:
                param_group['lr'] = d_lr

    def reset_grad(self):
        """Reset the gradient buffers."""
        self.g_optimizer.zero_grad()
        if self.learned_kernel or self.learned_alpha:
            self.d_optimizer.zero_grad()


    def label2onehot(self, labels, dim):
        """Convert label indices to one-hot vectors."""
        out = torch.zeros(list(labels.size())+[dim]).to(self.device)
        out.scatter_(len(out.size())-1,labels.unsqueeze(-1),1.)
        return out



    def sample_z(self, batch_size):
        return np.random.normal(0, 1, size=(batch_size, self.z_dim))

    def postprocess(self, inputs, method, temperature=1.):

        def listify(x):
            return x if type(x) == list or type(x) == tuple else [x]

        def delistify(x):
            return x if len(x) > 1 else x[0]

        if method == 'soft_gumbel':
            softmax = [F.gumbel_softmax(e_logits.contiguous().view(-1,e_logits.size(-1))
                       / temperature, hard=False).view(e_logits.size())
                       for e_logits in listify(inputs)]
        elif method == 'hard_gumbel':
            softmax = [F.gumbel_softmax(e_logits.contiguous().view(-1,e_logits.size(-1))
                       / temperature, hard=True).view(e_logits.size())
                       for e_logits in listify(inputs)]
        else:
            softmax = [F.softmax(e_logits / temperature, -1)
                       for e_logits in listify(inputs)]

        return [delistify(e) for e in (softmax)]

    def Compute_var(self,ry,yy,alpha):
        tensor_L = torch.exp(- (1 / (2 * alpha)) * (ry.t() + ry - 2 * yy))
        L2 = tensor_L.tolist()
        del (tensor_L)
        mylist = []
        for i in range(len(L2)):
            for j in range(i + 1, len(L2)):
                if i != j:
                    mylist.append(L2[i][j])
        my_a = np.array(mylist)
        my_var = np.var(my_a)
        print("variance for alpha "+str(alpha)+" is " +str(my_var))
        return my_var
    def best_alpha_range(self,ry,yy,start_range,end_range,start_value,end_value,stop=0.0001):
        if end_range-start_range<=stop:
            if start_value>=end_value:
                return start_range,start_value
            return end_range,end_value
        middle_alpha=(start_range+end_range)/2
        middle_value=self.Compute_var(ry,yy,middle_alpha)
        if middle_value<start_value and start_value>=end_value:
            return self.best_alpha_range(ry,yy,start_range,middle_alpha,start_value,middle_value)
        if middle_value < end_value and end_value > start_value:
            return self.best_alpha_range( ry, yy, middle_alpha, end_range, middle_value, end_value)
        best_alpha_start,best_value_start=self.best_alpha_range(ry,yy,start_range,middle_alpha,start_value,middle_value)
        best_alpha_end,best_value_end=self.best_alpha_range( ry, yy, middle_alpha, end_range, middle_value, end_value)
        if best_value_start >= best_value_end:
            return best_alpha_start, best_value_start
        return best_alpha_end, best_value_end

    def best_alpha(self,ry,yy):
        start = time.time()
        start_range=self.alpha/2
        if start_range<0.0001:
            start_range=0.0001
        start_value=self.Compute_var(ry,yy,start_range)
        end_range=self.alpha*2
        end_value=self.Compute_var(ry,yy,end_range)
        best_alpha,best_value=self.best_alpha_range(ry,yy,start_range,end_range,start_value,end_value)
        self.alpha=best_alpha
        end = time.time()
        print(str(end-start) +" seconds, best alpha: "+str(best_alpha)+" best alpha value: "+str(best_value))
        #raise Exception("hey")

    def MMD_mul(self,x, y):
        xx, yy, zz = torch.mm(x, x.t()), torch.mm(y, y.t()), torch.mm(x, y.t())
        rx = (xx.diag().unsqueeze(0).expand_as(xx))
        ry = (yy.diag().unsqueeze(0).expand_as(yy))

        dxx = rx.t() + rx - 2. * xx  # Used for A in (1)
        dyy = ry.t() + ry - 2. * yy  # Used for B in (1)
        dxy = rx.t() + ry - 2. * zz  # Used for C in (1)
        XX, YY, XY = (torch.zeros(xx.shape).to(self.device),
                      torch.zeros(xx.shape).to(self.device),
                      torch.zeros(xx.shape).to(self.device))
        for a in self.alphas:
            XX += torch.exp(-0.5 * dxx / a)
            YY += torch.exp(-0.5 * dyy / a)
            XY += torch.exp(-0.5 * dxy / a)

        return torch.mean(XX + YY - 2. * XY)
    def MMD_loss(self,x,y,compute_alpha=False,no_norm=False,alpha=False,Learned_alpha_model=False,multi_alpha=False):
        if not no_norm:
            x=torch.div(x, torch.norm(x,dim=1).unsqueeze_(1))
            y=torch.div(y, torch.norm(y,dim=1).unsqueeze_(1))
        if multi_alpha:
            return self.MMD_mul(x,y)
        B=x.size(0)
        xx, yy, zz = torch.mm(x,x.t()), torch.mm(y,y.t()), torch.mm(x,y.t())
        rx = (xx.diag().unsqueeze(0).expand_as(xx))
        ry = (yy.diag().unsqueeze(0).expand_as(yy))
        if compute_alpha:
            self.best_alpha(ry,yy)
        if not alpha:
            alpha =self.alpha
        if Learned_alpha_model:
            K = torch.exp(- Learned_alpha_model(rx.t() + rx - 2 * xx))
            L = torch.exp(- Learned_alpha_model(ry.t() + ry - 2 * yy))
            P = torch.exp(- Learned_alpha_model(rx.t() + ry - 2 * zz))
        else:
            K = torch.exp(- (1/(2*alpha)) * (rx.t() + rx - 2*xx))
            L = torch.exp(- (1/(2*alpha)) * (ry.t() + ry - 2*yy))
            P = torch.exp(- (1/(2*alpha)) * (rx.t() + ry - 2*zz))
        beta = (1./(B*B))
        gamma = (2./(B*B)) 
        return beta * (torch.sum(K)+torch.sum(L)) - gamma * torch.sum(P)


    def laplacian_func(self,edges):
        mult_matrix = torch.ones(5).to(self.device)
        mult_matrix[0] = 0
        edges=edges*mult_matrix
        edges= torch.sum(edges, -1)
        sum=torch.sum(edges,-1)
        lap=torch.diag_embed(sum)-edges
        sum=sum+0.000000001
        diag_sqrt=torch.diag_embed(torch.pow(sum,-0.5))
        ans=torch.bmm(lap,diag_sqrt)
        ans=torch.bmm(diag_sqrt, ans)
        ans = 0.5*(ans + ans.transpose(1,2))
        ans = torch.linalg.eigvals(ans)
        ans=torch.real(ans)
        ans,_=torch.sort(ans)
        if self.learned_laplacian:
            ans=self.Learned_laplacian_model(ans)
        return ans
    def get_mean_dev(self,a):
        diag = torch.diagonal(a,dim1=1,dim2=2)
        mean=torch.mean(diag, 1, True)
        std=torch.std(diag, 1, unbiased=False, keepdim=True)
        ans=torch.cat((std,mean),1)
        return ans
    def shapes_func(self,edges):
        mult_matrix = torch.ones(5).to(self.device)
        mult_matrix[0] = 0
        edges=edges*mult_matrix
        edges= torch.sum(edges, -1)
        a3=torch.linalg.matrix_power(edges, 3)
        ans=self.get_mean_dev(a3)
        for  i in range(4):
            a3=torch.bmm(a3,edges)
            ans2=self.get_mean_dev(a3)
            ans=torch.cat((ans,ans2),1)
        if(self.learned_shapes):
            ans=self.Learned_shapes_model(ans)
        return ans
    def train(self):

        # Learning rate cache for decaying.
        g_lr = self.g_lr
        d_lr = self.d_lr

        # Start training from scratch or resume training.
        start_iters = 0


        # Start training.
        print('Start training now...')
        start_time = time.time()
        disc_trained=0
        total_steps=0
        total_average_valid=0
        total_average_valid_unique=0
        gen_step=0
        self.G.train()
        self.Dtrain()
        for i in range(start_iters, self.num_iters):

            real_mols, _, _, a, x, _, _, _, _ = self.data.next_train_batch(self.batch_size)
            z = self.sample_z(self.batch_size)
            z_disc = self.sample_z(self.batch_size)

            # =================================================================================== #
            #                             1. Preprocess input data                                #
            # =================================================================================== #
            a = torch.from_numpy(a).to(self.device).long()            # Adjacency.
            x = torch.from_numpy(x).to(self.device).long()            # Nodes.
            a_tensor = self.label2onehot(a, self.b_dim)
            x_tensor = self.label2onehot(x, self.m_dim)
            z = torch.from_numpy(z).to(self.device).float()
            # =================================================================================== #
            #                             2. Train the Learned kernel                              #
            # =================================================================================== #
            loss = {}
            if self.learned_kernel:
                disc_trained=disc_trained+1
                z_disc = self.sample_z(self.batch_size)
                z_disc = torch.from_numpy(z_disc).to(self.device).float()
                self.d_optimizer.zero_grad()
                self.reset_grad()
                edges_logits, nodes_logits = self.G(z_disc)
                # Postprocess with Gumbel softmax
                (edges_hat, nodes_hat) = self.postprocess((edges_logits, nodes_logits), self.post_method)
                Dloss=0
                for j in range (self.num_learned_kernel):
                    logits_fake, features_fake = self.D[j](edges_hat, None, nodes_hat)
                    logits_real, features_real = self.D[j](a_tensor, None, x_tensor)
                    Dloss = Dloss -self.MMD_loss(features_fake, features_real,no_norm=True,Learned_alpha_model=self.Learned_alpha_model,multi_alpha=self.multi_alpha)
                if (self.laplacian > 0 and self.learned_laplacian):
                    laplacian_fet_real = self.laplacian_func(a_tensor)
                    laplacian_fet_fake = self.laplacian_func(edges_hat)
                    Dloss = Dloss-self.laplacian*self.MMD_loss(laplacian_fet_fake, laplacian_fet_real, compute_alpha=False, no_norm=True,
                                             alpha=self.laplacian_alpha)
                if (self.shapes > 0 and self.learned_shapes ):
                    shapes_fet_real = self.shapes_func(a_tensor)
                    shapes_fet_fake = self.shapes_func(edges_hat)
                    max_real, _ = torch.max(shapes_fet_real, dim=0)
                    min_real, _ = torch.min(shapes_fet_real, dim=0)
                    shapes_fet_real = (shapes_fet_real - min_real) / (max_real - min_real)
                    shapes_fet_fake = (shapes_fet_fake - min_real) / (max_real - min_real)
                    Dloss = Dloss-self.shapes*self.MMD_loss(shapes_fet_fake, shapes_fet_real, compute_alpha=False, no_norm=True,
                                                alpha=self.shapes_alpha)
                Dloss.backward()
                self.d_optimizer.step()
                loss['D/loss_value'] = Dloss

            # =================================================================================== #
            #                               3. Train the generator                                #
            # =================================================================================== #
            # Z-to-target
            edges_logits, nodes_logits = self.G(z)
            if ( not self.learned_kernel) or ( loss['D/loss_value'].item()<(self.min_loss*self.num_learned_kernel) and disc_trained>=self.n_critic):
                disc_trained=0
                gen_step=gen_step+1
                self.g_optimizer.zero_grad()
                # Postprocess with Gumbel softmax
                (edges_hat, nodes_hat) = self.postprocess((edges_logits, nodes_logits), self.post_method)
                gloss=0
                for j in range(self.num_disc):
                    if self.learned_kernel and j==0 and self.change_alpha>0 and gen_step%self.change_alpha==0:
                        compute_alpha=True
                    else:
                        compute_alpha=False
                    logits_fake, features_fake = self.D[j](edges_hat, None, nodes_hat)
                    logits_real, features_real = self.D[j](a_tensor, None, x_tensor)
                    gloss=gloss+self.MMD_loss(features_fake,features_real,compute_alpha=compute_alpha,no_norm=True,Learned_alpha_model=self.Learned_alpha_model)

                lap_loss=0
                if(self.laplacian>0):
                    laplacian_fet_real=self.laplacian_func(a_tensor)
                    laplacian_fet_fake=self.laplacian_func(edges_hat)
                    lap_loss =  self.MMD_loss(laplacian_fet_fake, laplacian_fet_real,compute_alpha=False,no_norm=True,alpha=self.laplacian_alpha)
                    loss['G/lap_loss_value'] = self.laplacian*lap_loss
                shapes_loss=0
                if(self.shapes>0):
                    shapes_fet_real=self.shapes_func(a_tensor)
                    shapes_fet_fake=self.shapes_func(edges_hat)
                    max_real,_=torch.max(shapes_fet_real,dim=0)
                    min_real,_=torch.min(shapes_fet_real,dim=0)
                    shapes_fet_real=(shapes_fet_real-min_real)/(max_real-min_real)
                    
                    shapes_fet_fake=(shapes_fet_fake-min_real)/(max_real-min_real)

                    shapes_loss =  self.MMD_loss(shapes_fet_fake, shapes_fet_real,compute_alpha=False,no_norm=True,alpha=self.shapes_alpha)
                    loss['G/shapes_loss'] = self.shapes*shapes_loss
                g_real_loss=gloss/self.num_disc+self.laplacian*lap_loss+self.shapes*shapes_loss
                g_real_loss.backward()
                self.g_optimizer.step()
                self.reset_grad()


                # Logging.
                loss['G/loss_value'] = gloss/self.num_disc




            # =================================================================================== #
            #                                 4. Miscellaneous                                    #
            # =================================================================================== #
            if (i) % self.samples_iter == 0 or (i+1) % self.log_step == 0:
                (edges_hard, nodes_hard) = self.postprocess((edges_logits, nodes_logits), 'hard_gumbel')
                edges_hard, nodes_hard = torch.max(edges_hard, -1)[1], torch.max(nodes_hard, -1)[1]
                mols = [self.data.matrices2mol(n_.data.cpu().numpy(), e_.data.cpu().numpy(), strict=True)
                        for e_, n_ in zip(edges_hard, nodes_hard)]
            # Print out training information.
            if (i) % self.samples_iter==0:
                filename=self.samples+"/"+str(i)+'-'+'fake.png'
                real_filename=self.real_samples+"/"+str(i)+'-'+'real.png'
                save_mol_img(mols, f_name=filename)
                save_mol_img(real_mols, f_name=real_filename)
            if (i+1) % self.log_step == 0:
                for j in range(self.num_learned_kernel,self.num_replacing_kernel+self.num_learned_kernel):
                    self.D[j] = Discriminator(self.d_conv_dim, self.m_dim, self.b_dim, self.dropout,self.device,self.random_graph_increase).to(self.device)
                real_log = {}
                et = time.time() - start_time
                et = str(datetime.timedelta(seconds=et))[:-7]
                log = "Elapsed [{}], Iteration [{}/{}]".format(et, i+1, self.num_iters)
                for tag, value in loss.items():
                    real_log[tag]=value.item()
                m0, m1 = all_scores(mols, self.data, norm=True)     # 'mols' is output of Fake Reward
                m0 = {k: np.array(v)[np.nonzero(v)].mean() for k, v in m0.items()}
                m0.update(m1)
                loss.update(m0)
                good_mols,bad_mols=try_create_mols(mols)
                self.good_mols.append(good_mols)
                self.bad_mols.append(bad_mols)
                total_steps = total_steps+1
                total_average_valid = total_average_valid+loss['valid score']
                real_log['average valid']=total_average_valid/total_steps
                self.valid50.append(loss['valid score'])
                real_log['average vailid last 50']=np.sum(self.valid50)/len(self.valid50)
                real_log.update(m0)
                if(self.Learned_alpha_model):
                    myalpha=self.Learned_alpha_model.get_alpha()
                else:
                    myalpha=self.alpha
                real_log['alpha']=myalpha
                real_log['unique valid score']=real_log['unique score']*real_log['valid score']/100
                self.unique_valid50.append(real_log['unique valid score'])
                total_average_valid_unique = total_average_valid_unique+real_log['unique valid score']
                real_log['average valid unique']=total_average_valid_unique/total_steps
                real_log['average valid unique last 50'] = np.sum(self.unique_valid50) / len(self.unique_valid50)
                for tag, value in real_log.items():
                     log += ", {}: {:.4f}".format(tag, value)
                print(log)
                with open(self.print_file, "a") as print_file:
                    print_file.write(log+"\n")
            # Save model checkpoints.
            if (i+1) % self.model_save_step == 0:
                G_path = os.path.join(self.model_save_dir, '{}-G.ckpt'.format(i+1))
                torch.save(self.G.state_dict(), G_path)
                print('Saved model checkpoints into {}...'.format(self.model_save_dir))

            # Decay learning rates.
            if (i+1) % self.lr_update_step == 0 and (i+1) > (self.num_iters - self.num_iters_decay):
                g_lr -= (self.g_lr / float(self.num_iters_decay))
                d_lr -= (self.d_lr / float(self.num_iters_decay))
                self.update_lr(g_lr, d_lr)
                print ('Decayed learning rates, g_lr: {}, d_lr: {}.'.format(g_lr, d_lr))





