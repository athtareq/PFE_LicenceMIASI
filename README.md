# PFE_LicenceMIASI
My undergraduate end of studies project / Mon projet de fin d'etudes de Licence. This is an unrevised version that I didn't submit to my supervisor, 
but I'm pretty sure the amount of errors is minimal (well at least I hope so). This document is only here on the off-chance someone is looking for specific elements
that were treated here, it merely serves as a support. I use a great deal of info that I find on github repos, that's why I'm sharing whatever things I do myself.  
# Notes
This project consisted of aggregating info to form what you can call a pseudo-course. I do not claim to have invented any of the featured theorems, methods or algorithms. I merely regrouped, reworded and organised whatever informations I thought would be useful to introduce the subject at hand. I probably used more than 100 resources, and I'm still trying to locate every single one, which I will add later on.   

Bibliography is at the end of the document

I used this beautiful template https://github.com/johannesbottcher/MDT-Quick-Manual/ 



# Abstract
***Differential equations are an important aspect of mathematical models and problems, hence
the ongoing need to find the appropriate tools to solve them. The subject matter of this project
is DeepXDE, a deep learning library developed in Brown University that can solve different
types of differential equations using neural networks. While the usage of our subject library is
only a matter of defining of the problem using TensorFlow syntax, understanding its learning
process would still require extensive knowledge of machine learning. Therefore we will give
an overview for differential equations and their important role in the mathematical modeling
of real life phenomenons, then we shall introduce key aspects of neural networks and deep
learning, before engaging in the presentation of the library, its functionalities and its
applications.***


# Chapter 1: On differential systems
    2.1 Ordinary differential equations (ODEs) 

        2.1.1 Background: 
    
	    2.1.2 Definitions: 
     
	    2.1.3 Classification: 
    
    2.2 Existence and uniqueness theory . . . . . . . . . . . . . . . . . .

    2.3 Periodic solutions to ODEs . . . . . . . . . . . . . . . . . . . . . .

    2.4 Boundary value problem . . . . . . . . . . . . . . . . . . . . . . .

    2.5 Linear differential systems . . . . . . . . . . . . . . . . . . . . . .

  	    Reduction of higher order systems to first order systems

    2.6 Solving differential equations . . . . . . . . . . . . . . . . . . . .

	    2.6.1 Euler’s method . . . . . . . . . . . . . . . . . . . . . . . .

			Python implementation . . . . . . . . . . . . . . . . . . .

	    2.6.2 Runge-Kutta method (RK4) . . . . . . . . . . . . . . . . .

			Python implementation . . . . . . . . . . . . . . . . . . .

# Chapter 2: Neural Networks and Deep Learning

    3.1 Introduction . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .

	    3.1.1 Analogy with the human brain . . . . . . . . . . . . . . . . .

    3.2 Artificial Neural Networks . . . . . . . . . . . . . . . . . . . . . . . .

        Understanding the artificial neuron . . . . . . . . . . . . . .

    3.3 The learning process . . . . . . . . . . . . . . . . . . . . . . . . . . .

         Key definitions . . . . . . . . . . . . . . . . . . . . . . . . . .
    
         Activation functions . . . . . . . . . . . . . . . . . . . . . . .
    
         Gradient descent optimization algorithms . . . . . . . . . .
    
         The formal process of learning . . . . . . . . . . . . . . . . .
    
         The backpropagation algorithm . . . . . . . . . . . . . . . .
    
         Python implementation of the training of a neural network .

    3.4 Approximation capabilities of an artificial neural network . . . . .
    
		George Cybenko, 1989 . . . . . . . . . . . . . . . . . . . . . .
    
		Kurt Hornik, 1991 . . . . . . . . . . . . . . . . . . . . . . . . .

# Chapter 3: DeepXDE 

    4.1 Automatic Differentiation . . . . . . . . . . . . . . . . . .

        4.1.1 Definition and comparison with backpropagation

        4.1.2 The process . . . . . . . . . . . . . . . . . . . . . .

             Forward mode . . . . . . . . . . . . . . . . . . . .
    
             Reverse mode . . . . . . . . . . . . . . . . . . . . . .
    
    4.2 Physics-Informed Neural Networks (PINNs) . . . . . . . . . . . . . . . . . . . . . 
    
        4.2.1 The concept of PINNs . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 
    
        4.2.2 PINNs for solving partial differential equations . . . . . . . . . . . . . . . 
      
              Step 1: Constructing the neural network. . . . . . . . . . . . . . . . . . . . 
      
              Step 2: Specifying the two training sets for the equation and boundary/initial conditions . . . . . . . . . . . . . . . . . 
      
              Step 3: Compute the difference between the neural network and the constraints . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 
      
              Step 4: Training the neural network to find the best parameters by minimizing the loss function . . . . . . . . . . . . . . . . . . . . . . . . 

    4.3 Adam optimization . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 

    4.4 Usage of DeepXDE . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
    
		4.4.1 Installation . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 
    
		4.4.2 Procedure . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 
    
			  1.Geometry . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .    
      
			  2.Defining the differential problem . . . . . . . . . . . . . . . . . . . . . . . 
      
			  3.Specifying the boundary/initial conditions . . . . . . . . . . . . . . . . . 
			
			  4.The data module . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 
			
			  Constructing the neural network . . . . . . . . . . . . . . . . . . . . . . . . 

			  Defining the model and compiling it . . . . . . . . . . . . . . . . . . . . . . 
              
              
              
# Chapter 4: Applications

	5.1 ODE system . . . . . . . . . . . . . . . . . . . . .

	5.2 Periodic solution . . . . . . . . . . . . . . . . . .

	5.3 Non linear equation . . . . . . . . . . . . . . . . .

	5.4 2D equation on a rectangle . . . . . . . . . . . . .

	5.5 Modeling of a real life phenomenon: COVID-19

		5.5.1 The problem’s settings . . . . . . . . . . .
