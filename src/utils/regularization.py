import torch

def generate_random_points(domain, num_points):
    """
    Generate random points in the given domain.
    
    Parameters:
        domain (list of tuples): [(x_min, x_max), (y_min, y_max)]
        num_points (int): Number of random points to generate.
        
    Returns:
        torch.Tensor: Random points in the domain.
    """
    x_min, x_max = domain[0]
    y_min, y_max = domain[1]
    x = torch.FloatTensor(num_points, 1).uniform_(x_min, x_max)
    y = torch.FloatTensor(num_points, 1).uniform_(y_min, y_max)
    return torch.cat([x, y], dim=1)

def sample_unit_square_points(center, sl=0.1, num_points_per_side=10):
    """
    Sample points on the unit square centered around a given point.
    
    Parameters:
        center (torch.Tensor): The center of the square, a tensor of shape (2,).
        sl (float): The side length of the square.
        num_points_per_side (int): The number of points to sample on each side of the square.
        
    Returns:
        torch.Tensor: A tensor containing the sampled points of shape (4 * num_points_per_side, 2).
    """
    half_sl = sl / 2.0
    x_center, y_center = center
    
    # Generate points on each side of the square
    x_coords = torch.linspace(x_center - half_sl, x_center + half_sl, num_points_per_side)
    y_coords = torch.linspace(y_center - half_sl, y_center + half_sl, num_points_per_side)

    # Top side (from left to right)
    top_side = torch.stack([x_coords, torch.full((num_points_per_side,), y_center + half_sl)], dim=1)
    
    # Bottom side (from left to right)
    bottom_side = torch.stack([x_coords, torch.full((num_points_per_side,), y_center - half_sl)], dim=1)
    
    # Left side (from bottom to top)
    left_side = torch.stack([torch.full((num_points_per_side,), x_center - half_sl), y_coords], dim=1)
    
    # Right side (from bottom to top)
    right_side = torch.stack([torch.full((num_points_per_side,), x_center + half_sl), y_coords], dim=1)

    # Combine all the points
    square_points = torch.cat([top_side, bottom_side, left_side, right_side], dim=0)
    
    return square_points

def cons_vol_loss(pred_x):
    """
    Compute volume conservation loss based on Liouville's theorem.
    Measures how well the model preserves volume in phase space.
    """
    q1_bounds = torch.rand(2).sort()[0] - 0.5
    p1_bounds = torch.rand(2).sort()[0]*0.2 - 0.1
    q2_bounds = torch.rand(2).sort()[0]*0.2 - 0.1
    p2_bounds = torch.rand(2).sort()[0]*0.2 - 0.1
    in_domain = torch.cat([pred_x[:,:,:,0:1] > q1_bounds[0], 
                           pred_x[:,:,:,0:1] < q1_bounds[1], 
                           pred_x[:,:,:,1:2] > q2_bounds[0], 
                           pred_x[:,:,:,1:2] < q2_bounds[1], 
                           pred_x[:,:,:,2:3] > p1_bounds[0], 
                           pred_x[:,:,:,2:3] < p1_bounds[1], 
                           pred_x[:,:,:,3:4] > p2_bounds[0], 
                           pred_x[:,:,:,3:4] < p2_bounds[1]], dim=-1)
    in_domain = torch.all(in_domain, dim=-1).squeeze(-1)
    in_domain = in_domain.type(torch.Tensor)
    in_domain = torch.mean(in_domain, dim=1)
    return torch.mean((in_domain - in_domain[0])**2)

def compute_energy_conservation_loss(model, pred_x):
    """
    Compute energy conservation loss to ensure the Hamiltonian is preserved.
    """
    hamiltonians_0 = model.sample_hamiltonian(pred_x[0])
    
    energy_loss = 0
    for i in range(pred_x.shape[0]-1):
        energy_loss += torch.mean((model.sample_hamiltonian(pred_x[i+1]) - hamiltonians_0)**2)
    
    return energy_loss

def compute_hamiltonian_origin_penalty(model):
    """
    Penalize non-zero Hamiltonian at the origin.
    """
    zeros = torch.zeros(1, 1, model.d)  # Use model's input dimension
    hamiltonians_00 = model.sample_hamiltonian(zeros)
    return torch.abs(hamiltonians_00)

def compute_negative_hamiltonian_penalty(model, x):
    """
    Penalize negative Hamiltonian values.
    """
    hamiltonians = model.sample_hamiltonian(x)
    return torch.relu(-hamiltonians).mean()

def compute_loop_action_loss(model, batch_t, bounds, NL=1, Ntau=100):
    """
    Compute loop action loss to enforce symplectic structure preservation.
    """
    # Import torchdiffeq only when needed
    from torchdiffeq import odeint
    
    loss = torch.tensor(0.0, device=batch_t.device)
    tau = torch.linspace(0, 2*torch.pi, Ntau, device=loss.device)
    
    for l in range(NL):
        rand = torch.rand((5,3), device=loss.device) - 0.5
        q1 = (bounds[0,1]-bounds[0,0])*(rand[0,0] + rand[0,1]*torch.sin(tau + 2*torch.pi*rand[0,2])) + bounds[0,0]
        q2 = (bounds[1,1]-bounds[1,0])*(rand[1,0] + rand[1,1]*torch.sin(tau + 2*torch.pi*rand[1,2])) + bounds[1,0]
        p1 = (bounds[2,1]-bounds[2,0])*(rand[2,0] + rand[2,1]*torch.sin(tau + 2*torch.pi*rand[2,2])) + bounds[2,0]
        p2 = (bounds[3,1]-bounds[3,0])*(rand[3,0] + rand[3,1]*torch.sin(tau + 2*torch.pi*rand[3,2])) + bounds[3,0]
        t  = (bounds[4,1]-bounds[4,0])*(rand[4,0] + rand[4,1]*torch.sin(tau + 2*torch.pi*rand[4,2])) + bounds[4,0]
        
        batch_x0 = torch.cat((q1.reshape(-1,1), q2.reshape(-1,1), p1.reshape(-1,1), p2.reshape(-1,1)), dim=1)
        H = model.sample_hamiltonian(batch_x0).flatten()
        
        loop_action_0 = torch.tensor(0.0, device=batch_t.device)
        for i in range(len(tau)-1):
            loop_action_0 += p1[i]*(q1[i+1]-q1[i]) + p2[i]*(q2[i+1]-q2[i]) - H[i]*(t[i+1]-t[i])
        
        batch_x0 = batch_x0.reshape(batch_x0.shape[0], 1, -1)
        pred_xt = odeint(model, batch_x0, batch_t, method='fehlberg2', atol=1e-4, rtol=1e-4)
        
        for T in range(len(batch_t)):
            q1 = pred_xt[T,:,:,0].flatten()
            q2 = pred_xt[T,:,:,1].flatten()
            p1 = pred_xt[T,:,:,2].flatten()
            p2 = pred_xt[T,:,:,3].flatten()
            H = model.sample_hamiltonian(pred_xt[T]).flatten()
            
            loop_action_T = torch.tensor(0.0, device=batch_t.device)
            for i in range(len(tau)-1):
                loop_action_T += p1[i]*(q1[i+1]-q1[i]) + p2[i]*(q2[i+1]-q2[i]) - H[i]*(t[i+1]-t[i])
            
            loss += (loop_action_0 - loop_action_T)**2
    
    loss = loss / NL / len(batch_t)
    return loss