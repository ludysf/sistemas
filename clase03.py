# Estas son las librerías principales a utilizar 
import streamlit as st 
import numpy as np #librería operaciones numéricas 
import matplotlib.pyplot as plt #graficar
from scipy import signal
import math
import matplotlib.pyplot as plt
#st.set_option('deprecation.showPyplotGlobalUse', False)

st.title('CONVOLUCIÓN DE SEÑALES')
st.text('Python Version')



opp= st.sidebar.selectbox('Dominio del tiempo',['Selection','Continuo','Discreto'])


if opp=='Continuo':
   
   op= st.sidebar.selectbox('Señal de entrada x(t)',['Selection','a*sin(2*pi*f*t)','ae^-bt','Triangular','Rectangular','Señal rampa 1','Señal rampa 2','Señal rampa 3']) 
 
   if  op=='a*sin(2*pi*f*t)':
       a= st.sidebar.number_input('Select a:')
       b= st.sidebar.number_input('Select f:')
       c= st.sidebar.number_input('Select tlim inferior:' )
       d= st.sidebar.number_input('Select tlim superior:' )
       x= np.arange(c,d,0.001)
       r=0.001
       y= a*np.sin(2*np.pi*b*x)
    
    

   elif op=='Triangular':
       a = st.sidebar.number_input('Select amplitud: ')
       b = st.sidebar.number_input('Select f: ',1)
       c= st.sidebar.number_input('Select tlim inferior:' )
       d= st.sidebar.number_input('Select tlim superior:')
       x= np.arange(c,d,0.001)
       r=0.001
       y = a*signal.sawtooth(2*np.pi*b*x+(np.pi/2),0.5) 


   elif op == 'Rectangular':
       a = st.sidebar.number_input('Select amplitud: ')
       b = st.sidebar.number_input('Select f: ',1)
       c= st.sidebar.number_input('Select tlim inferior:', )
       d= st.sidebar.number_input('Select tlim superior:', )
       x= np.arange(c,d,0.001)
       r=0.001
       y = a*signal.square(2*np.pi*b*x)

   elif op == 'Señal rampa 1':
       a = st.sidebar.number_input('Select xlim inferior: ')
       b = st.sidebar.number_input('Select xlim superior: ')
       x = np.arange(a, b, 0.001)
       c=a
       d=b

       def u(x): 
          return np.piecewise(x,[x<0.0,x>=0.0],[0,1]) 
       y=(1.1*x-2.3)*(u(x-3)-u(x-6))-(u(x-3)-u(x-6))+3.3*u(x-6)
       r=0.001

   elif op == 'Señal rampa 2':
       a = st.sidebar.number_input('Select xlim inferior: ')
       b = st.sidebar.number_input('Select xlim superior: ')
       x = np.arange(a, b, 0.001)
       c=a
       d=b

       def u(x): 
          return np.piecewise(x,[x<0.0,x>=0.0],[0,1]) 
       y=3.3-3.3*u(x-4)-(1.1*(x-7)*u(x-4))+(1.1*(x-7)*u(x-7))
       r=0.001

   elif op == 'Señal rampa 3':
       a = st.sidebar.number_input('Select xlim inferior: ')
       b = st.sidebar.number_input('Select xlim superior: ')
       x = np.arange(a, b, 0.001)
       c=a
       d=b
       def u(x): 
          return np.piecewise(x,[x<0.0,x>=0.0],[0,1]) 
       y=1.1*x*u(x)-1.1*x*u(x-3)+3.3*u(x-3)-1.1*(x-7)*u(x-7)-(3.3*u(x-3)-1.1*(x-7)*u(x-7))*u(x-10)
       r=0.001
   else:
       a= st.sidebar.number_input('Select a:')
       b= st.sidebar.number_input('Select b:')
       c= st.sidebar.number_input('Select tlim inferior:' )
       d= st.sidebar.number_input('Select tlim superior:' )
       x= np.arange(c,d,0.001)
       y= a*np.exp(-b*x)
       r=0.001
   graph=st.sidebar.button('PLOT')

   if graph:
       st.header('Función de entrada x(t)')
       plt.figure(1,figsize=(10,5))
       plt.plot(x, y, 'r') # plotting t, a separately 
       plt.legend(['x(t)'])
       plt.ylabel('y')
       plt.xlabel('t')
       plt.title(op)
       plt.grid(True)
       st.pyplot()
       st.pyplot(clear_figure=True)
       plt.show()



   op1= st.sidebar.selectbox('Respuesta al impulso h(t)',['Selection','a*sin(2*pi*f*t).','Ae^-bt.','Triangular.','Rectangular.','Señal rampa 1.','Señal rampa 2.','Señal rampa 3.'])
    #st.write(op1)#


   if  op1=='a*sin(2*pi*f*t).':
       ah= st.sidebar.number_input('Select a.:',)
       bh= st.sidebar.number_input('Select f.:')
       ch= st.sidebar.number_input('Select tlim inferior.:' )
       dh= st.sidebar.number_input('Select tlim superior.:' )
       xh= np.arange(ch,dh,0.001)
       yh= ah*np.sin(2*np.pi*bh*xh)
       yhflip=ah*np.sin(2*np.pi*bh*-xh)
       r=0.001

   elif op1=='Triangular.':
       ah = st.sidebar.number_input('Select amplitud.: ')
       bh = st.sidebar.number_input('Select f.: ')
       ch= st.sidebar.number_input('Select tlim inferior.:', )
       dh= st.sidebar.number_input('Select tlim superior.:', )
       xh= np.arange(ch,dh,0.001)
       yh = ah*signal.sawtooth(2*np.pi*bh*xh+(np.pi/2),0.5)  
       r=0.001
       yhflip=ah*signal.sawtooth(2*np.pi*-bh*xh+(np.pi/2),0.5) 


   elif op1 == 'Rectangular.':
       ah = st.sidebar.number_input('Select amplitud.: ')
       bh = st.sidebar.number_input('Select f.: ')
       ch= st.sidebar.number_input('Select tlim inferior.:', )
       dh= st.sidebar.number_input('Select tlim superior.:', )
       xh= np.arange(ch,dh,0.001)
       yh = ah*signal.square(2*np.pi*bh*xh)
       r=0.001
       yhflip=ah*signal.square(2*np.pi*bh*-xh)

   elif op1 == 'Señal rampa 1.':
       ah = st.sidebar.number_input('Select xlim inferior.: ')
       bh = st.sidebar.number_input('Select xlim superior.: ')
       xh = np.arange(ah, bh, 0.001)
       ch=ah
       dh=bh

       def u(xh): 
          return np.piecewise(xh,[xh<0.0,xh>=0.0],[0,1]) 
       yh=(1.1*xh-2.3)*(u(xh-3)-u(xh-6))-(u(xh-3)-u(xh-6))+3.3*u(xh-6)
       def uflip(xh): 
          return np.piecewise(xh,[xh>0.0,xh<=0.0],[0,1]) 
       
       yhflip=(1.1*xh-2.3)*(uflip(xh-3)-uflip(xh-6))-(uflip(xh-3)-uflip(xh-6))+3.3*uflip(xh-6)
       r=0.001

   elif op1 == 'Señal rampa 2.':
       ah = st.sidebar.number_input('Select xlim inferior.: ')
       bh = st.sidebar.number_input('Select xlim superior.: ')
       xh= np.arange(ah, bh, 0.001)
       ch=ah
       dh=bh

       def u(xh): 
          return np.piecewise(xh,[xh<0.0,xh>=0.0],[0,1]) 
       yh=3.3-3.3*u(xh-4)-(1.1*(xh-7)*u(xh-4))+(1.1*(xh-7)*u(xh-7))
       def uflip(xh): 
          return np.piecewise(xh,[xh>0.0,xh<=0.0],[0,1]) 
       yhflip=3.3-3.3*uflip(xh-4)-(1.1*(xh-7)*uflip(xh-4))+(1.1*(xh-7)*uflip(xh-7))
       r=0.001

   elif op1=='Señal rampa 3.':
       ah = st.sidebar.number_input('Select xlim inferior.: ')
       bh = st.sidebar.number_input('Select xlim superior.: ')
       xh = np.arange(ah, bh, 0.001)
       ch=ah
       dh=bh
       def u(xh): 
          return np.piecewise(xh,[xh<0.0,xh>=0.0],[0,1]) 
       yh=1.1*xh*u(xh)-1.1*xh*u(xh-3)+3.3*u(xh-3)-1.1*(xh-7)*u(xh-7)-(3.3*u(xh-3)-1.1*(xh-7)*u(xh-7))*u(xh-10)  
       def uflip(xh): 
          return np.piecewise(xh,[xh>0.0,xh<=0.0],[0,1])
       yhflip=yh  
       r=0.001
   else:
       
       ah= st.sidebar.number_input('Select a.:')
       bh= st.sidebar.number_input('Select b.:')
       ch= st.sidebar.number_input('Select tlim inferior.:' )
       dh= st.sidebar.number_input('Select tlim superior.:' )
       xh= np.arange(ch,dh,0.001)
       yh= ah*np.exp(-bh*xh)
       yhflip=ah*np.exp(bh*xh)
       r=0.001
    
   graph1=st.sidebar.button('PLOT2')

   if graph1:
       st.header('Función de entrada x(t)')
       plt.figure(1,figsize=(10,5))
       plt.plot(x, y, 'r') # plotting t, a separately 
       plt.legend(['x(t)'])
       plt.ylabel('y')
       plt.xlabel('t')
       plt.title(op)
       plt.grid(True)
       st.pyplot()
    

       st.header('Respuesta al impulso h(t)')
       plt.figure(2,figsize=(10,5))
       plt.plot(xh, yh, 'b') # plotting t, b separately 
       plt.legend(['h(t)'])
       plt.ylabel('h')
       plt.xlabel('t')
       plt.title(op1)
       plt.grid(True)
       st.pyplot()
       plt.show()
       #convolución para el continuo
  
   covolu=st.sidebar.button('CONVOLUCIÓN')
       
   if covolu:
       st.header('Función de entrada x(t)')
       plt.figure(1,figsize=(10,5))
       plt.plot(x, y, 'r') # plotting t, a separately 
       plt.legend(['x(t)'])
       plt.ylabel('y')
       plt.xlabel('t')
       plt.title(op)
       plt.grid(True)
       st.pyplot()
    

       st.header('Respuesta al impulso h(t)')
       plt.figure(2,figsize=(10,5))
       plt.plot(xh, yh, 'b',) # plotting t, b separately 
       plt.legend(['h(t)'])
       plt.ylabel('h')
       plt.xlabel('t')
       plt.title(op1)
       plt.grid(True)
       st.pyplot()
       plt.show()

       st.header('Convolución y(t)')
       
       L=len(x)
       M=len(xh)
       Y = L+M-1
       tyc = np.arange(c+ch, d+dh-r, r)
      
       yc = r*np.convolve(y,yh)
       hs=yhflip
       
       taux = xh - (d-c)
       step = 200 # Velocidad de animación
       frames = 2* (d-c)/Y*step

       bar = st.empty()
       graph_xh = st.empty()
       graph_yc = st.empty()

       for i in range (0,L + M - 1, step):
          bar.progress(i/Y)
 
          
          plt.plot(x,y,'r')
          plt.plot(taux,hs)
          plt.legend(['x(T)','h(t-T)'])
          plt.xlim(c+ch-dh, d+dh+dh)
          plt.title('Desplazamiento de h(t-T) sobre x(T)')
          graph_xh.pyplot()

          plt.plot(tyc[:i],yc[:i],"orange")
          plt.legend(['y(t)'])
          plt.title('CONVOLUCIÓN')
          plt.xlim(c+ch, d+dh)
          plt.ylim(min(yc)-1, max(yc)+1)
          graph_yc.pyplot()

          taux = taux + frames
elif opp=='Discreto' :

   op= st.sidebar.selectbox('Señal de entrada x[n]',['Selection','a*sin(2*pi*f*n)','ae^-bn','Triangular','Rectangular','Señal rampa 1','Señal rampa 2','Señal rampa 3']) 
 
   if  op=='a*sin(2*pi*f*n)':
       a= st.sidebar.number_input('Select a:')
       b= st.sidebar.number_input('Select f:',0.01)
       c= st.sidebar.number_input('Select nlim inferior:' )
       d= st.sidebar.number_input('Select nlim superior:' )
       fs=20*b
       x= np.arange(c,d,(1/fs))
       y= a*np.sin(2*np.pi*b*x)
       x2= np.arange(c,c+len(y),1)
       d=c+len(y)
    
    
    
   elif op=='Triangular':
       a = st.sidebar.number_input('Select amplitud: ')
       b = st.sidebar.number_input('Select f: ',0.01)
       c= st.sidebar.number_input('Select tlim inferior:' )
       d= st.sidebar.number_input('Select tlim superior:')
       fs = 20*b
       x= np.arange(c,d,(1/fs))
       y = a*signal.sawtooth(2*np.pi*b*x+(np.pi/2),0.5) 
       x2 = np.arange(c,c+len(y),1)                                                                                                                                                
       d=c+len(y)


   elif op == 'Rectangular':
       a = st.sidebar.number_input('Select amplitud: ')
       b = st.sidebar.number_input('Select f: ',0.01)
       c= st.sidebar.number_input('Select tlim inferior:', )
       d= st.sidebar.number_input('Select tlim superior:', )
       fs = 20*b
       x= np.arange(c,d,(1/fs))
       y = a*signal.square(2*np.pi*b*x)
       x2 = np.arange(c,c+len(y),1)
       d=c+len(y)

   elif op == 'Señal rampa 1':
       a = st.sidebar.number_input('Select xlim inferior: ',0.01)
       b = st.sidebar.number_input('Select xlim superior: ',0.02)
       fs=20/(b-a)
       x = np.arange(a, b, (1/fs))
       c=a
       d=b

       def u(x): 
          return np.piecewise(x,[x<0.0,x>=0.0],[0,1]) 
       y=(1.1*x-2.3)*(u(x-3)-u(x-6))-(u(x-3)-u(x-6))+3.3*u(x-6)
       x2=np.arange(c,c+len(y),1)
       d=c+len(y)

   elif op == 'Señal rampa 2':
       a = st.sidebar.number_input('Select xlim inferior: ', 0.01)
       b = st.sidebar.number_input('Select xlim superior: ', 0.02)
       fs = 20/(b-a)
       x = np.arange(a, b, (1/fs))
       c=a
       d=b

       def u(x): 
          return np.piecewise(x,[x<0.0,x>=0.0],[0,1]) 
       y=3.3-3.3*u(x-4)-(1.1*(x-7)*u(x-4))+(1.1*(x-7)*u(x-7))
       r=0.001
       x2 = np.arange(c,c+len(y),1)
       d=c+len(y)

   elif op == 'Señal rampa 3':
       a = st.sidebar.number_input('Select xlim inferior: ',0.0)
       b = st.sidebar.number_input('Select xlim superior: ',0.01)
       fs = 20/(b-a)
       x = np.arange(a, b, (1/fs))
       c=a
       d=b
       def u(x): 
          return np.piecewise(x,[x<0.0,x>=0.0],[0,1]) 
       y=1.1*x*u(x)-1.1*x*u(x-3)+3.3*u(x-3)-1.1*(x-7)*u(x-7)-(3.3*u(x-3)-1.1*(x-7)*u(x-7))*u(x-10)
       x2 = np.arange(c,c+len(y),1)
       d=c+len(y) 

   else:
       a= st.sidebar.number_input('Select a:')
       b= st.sidebar.number_input('Select b:')
       c= st.sidebar.number_input('Select tlim inferior:', 0.01 )
       d= st.sidebar.number_input('Select tlim superior:', 0.02 )
       fs = 20/(d-c)
       x = np.arange(c, d, (1/fs))
       y= a*np.exp(-b*x)
       x2 = np.arange(c,c+len(y),1)
       d=c+len(y)

   graph=st.sidebar.button('PLOT')

   if graph:
       st.header('Función de entrada x[n]')
       plt.figure(1,figsize=(10,5))
       plt.stem(x2, y, 'r') # plotting t, a separately 
       plt.legend(['x[n]'])
       plt.ylabel('y')
       plt.xlabel('n')
       plt.title(op)
       plt.grid(True)
       st.pyplot()
       st.pyplot(clear_figure=True)
       plt.show()



   op1= st.sidebar.selectbox('Respuesta al impulso h(t)',['Selection','a*sin(2*pi*f*t).','Ae^-bt.','Triangular.','Rectangular.','Señal rampa 1.','Señal rampa 2.','Señal rampa 3.'])
    #st.write(op1)#


   if  op1=='a*sin(2*pi*f*t).':
       ah= st.sidebar.number_input('Select a.:')
       bh= st.sidebar.number_input('Select f.:',0.01)
       ch= st.sidebar.number_input('Select nlim inferior.:' )
       dh= st.sidebar.number_input('Select nlim superior.:' )
       fsh=20*bh
       xh= np.arange(ch,dh,(1/fsh))
       yh= ah*np.sin(2*np.pi*bh*xh)
       x2h= np.arange(ch,ch+len(yh),1)
       dh=ch+len(yh)
       
       yhflip=ah*np.sin(2*np.pi*bh*-xh)
       

   elif op1=='Triangular.':
       ah = st.sidebar.number_input('Select amplitud.: ')
       bh = st.sidebar.number_input('Select f.: ',0.01)
       ch= st.sidebar.number_input('Select tlim inferior.:' )
       dh= st.sidebar.number_input('Select tlim superior.:')
       fsh = 20*bh
       xh= np.arange(ch,dh,(1/fsh))
       yh = ah*signal.sawtooth(2*np.pi*bh*xh+(np.pi/2),0.5) 
       x2h = np.arange(ch,ch+len(yh),1)

       dh=ch+len(yh)
       yhflip=ah*signal.sawtooth(2*np.pi*bh*(-xh)+(np.pi/2),0.5) 


   elif op1 == 'Rectangular.':
       ah = st.sidebar.number_input('Select amplitud.: ')
       bh = st.sidebar.number_input('Select f.: ',0.01)
       ch= st.sidebar.number_input('Select tlim inferior.:', )
       dh= st.sidebar.number_input('Select tlim superior.:', )
       fsh = 20*bh
       xh= np.arange(ch,dh,(1/fsh))
       yh = ah*signal.square(2*np.pi*bh*xh)
       x2h = np.arange(ch,ch+len(yh),1)

       dh=ch+len(yh)
       yhflip=ah*signal.square(2*np.pi*bh*(-xh))

   elif op1 == 'Señal rampa 1.':
       ah = st.sidebar.number_input('Select xlim inferior.: ', 0.01)
       bh = st.sidebar.number_input('Select xlim superior.: ', 0.02)
       ch=ah
       dh=bh
       fsh = 20/(bh-ah)
       xh = np.arange(ah, bh, (1/fsh))
       

       def u(xh): 
          return np.piecewise(xh,[xh<0.0,xh>=0.0],[0,1]) 
       yh=(1.1*xh-2.3)*(u(xh-3)-u(xh-6))-(u(xh-3)-u(xh-6))+3.3*u(xh-6)
       def uflip(xh): 
          return np.piecewise(xh,[xh>0.0,xh<=0.0],[0,1]) 
       x2h = np.arange(ch,ch+len(yh),1)
       yhflip=(1.1*xh-2.3)*(uflip(xh-3)-uflip(xh-6))-(uflip(xh-3)-uflip(xh-6))+3.3*uflip(xh-6)
       dh=ch+len(yh)
       

       

   elif op1 == 'Señal rampa 2.':
       ah = st.sidebar.number_input('Select xlim inferior.: ', 0.01)
       bh = st.sidebar.number_input('Select xlim superior.: ', 0.02)
       fsh = 20/(bh-ah)
       xh= np.arange(ah, bh, (1/fsh))
       ch=ah
       dh=bh

       def u(xh): 
          return np.piecewise(xh,[xh<0.0,xh>=0.0],[0,1]) 
       yh=3.3-3.3*u(xh-4)-(1.1*(xh-7)*u(xh-4))+(1.1*(xh-7)*u(xh-7))
       def uflip(xh): 
          return np.piecewise(xh,[xh>0.0,xh<=0.0],[0,1]) 
       x2h = np.arange(ch,ch+len(yh),1)
       yhflip=3.3-3.3*uflip(xh-4)-(1.1*(xh-7)*uflip(xh-4))+(1.1*(xh-7)*uflip(xh-7))
       dh=ch+len(yh)

   elif op1=='Señal rampa 3.':
       ah = st.sidebar.number_input('Select xlim inferior.: ', 0.00)
       bh = st.sidebar.number_input('Select xlim superior.: ', 0.01)
       fsh = 20/(bh-ah)
       xh = np.arange(ah, bh, (1/fsh))
       ch=ah
       dh=bh
       def u(xh): 
          return np.piecewise(xh,[xh<0.0,xh>=0.0],[0,1]) 
       yh=1.1*xh*u(xh)-1.1*xh*u(xh-3)+3.3*u(xh-3)-1.1*(xh-7)*u(xh-7)-(3.3*u(xh-3)-1.1*(xh-7)*u(xh-7))*u(xh-10)  
       def uflip(xh): 
          return np.piecewise(xh,[xh>0.0,xh<=0.0],[0,1])
       x2h = np.arange(ch,ch+len(yh),1)
       yhflip=yh  
       dh=ch+len(yh)
       
   else:
       
       ah= st.sidebar.number_input('Select a.:')
       bh= st.sidebar.number_input('Select b.:')
       ch= st.sidebar.number_input('Select tlim inferior.:', 0.01 )
       dh= st.sidebar.number_input('Select tlim superior.:', 0.02 )
       fsh = 20/(dh-ch)
       xh = np.arange(ch, dh, (1/fsh))
       yh= ah*np.exp(-bh*xh)
       x2h = np.arange(ch,ch+len(yh),1)
       yhflip=ah*np.exp(bh*xh)
       dh=ch+len(yh)
    
   graph1=st.sidebar.button('PLOT2')

   if graph1:
       st.header('Función de entrada x[n]')
       plt.figure(1,figsize=(10,5))
       plt.stem(x2, y, 'r') # plotting t, a separately 
       plt.legend(['x[n]'])
       plt.ylabel('y')
       plt.xlabel('n')
       plt.title(op)
       plt.grid(True)
       st.pyplot()
    

       st.header('Respuesta al impulso h[n]')
       plt.figure(2,figsize=(10,5))
       plt.stem(x2h, yh, 'b') # plotting t, b separately 
       plt.legend(['h[n]'])
       plt.ylabel('h')
       plt.xlabel('n')
       plt.title(op1)
       plt.grid(True)
       st.pyplot()
       plt.show()
       #convolución para el discreto
  
   covolu=st.sidebar.button('CONVOLUCIÓN')
       
   if covolu:
       st.header('Función de entrada x[n]')
       plt.figure(1,figsize=(10,5))
       plt.stem(x2, y, 'r') # plotting t, a separately 
       plt.legend(['x[n]'])
       plt.ylabel('y')
       plt.xlabel('n')
       plt.title(op)
       plt.grid(True)
       st.pyplot()
    

       st.header('Respuesta al impulso h[n]')
       plt.figure(2,figsize=(10,5))
       plt.stem(x2h, yh, 'b',) # plotting t, b separately 
       plt.legend(['h[n]'])
       plt.ylabel('h')
       plt.xlabel('n')
       plt.title(op1)
       plt.grid(True)
       st.pyplot()
       plt.show()

       st.header('Convolución y[n]')
       
       L=len(y)
       M=len(yh)
       Y = L+M-1
       tyc = np.arange(c+ch, d+dh-1, 1)
       tycf = list()
       yc = np.convolve(y,yh)
       ycf=list()
       hs=yhflip
       
       taux = x2h - (d-c)
       step = 1 # Velocidad de animación
       frames = 2* (d-c)/Y*step

       bar = st.empty()
       graph_xh = st.empty()
       graph_yc = st.empty()

       for i in range (0,L + M - 1, step):
          bar.progress(i/Y)
          ycf.append(yc[i])
          tycf.append(tyc[i])
          plt.stem(x2,y,'r')
          plt.stem(taux,hs)
          plt.legend(['x[k]','h[n-k]'])
          plt.xlim(c+ch-dh, d+dh+dh)
          plt.title('Desplazamiento de h[n-k] sobre x[k]')
          graph_xh.pyplot()

          plt.stem(tycf,ycf,basefmt="limegreen",linefmt="darkorange")
          plt.legend(['y[n]'])
          plt.title('CONVOLUCIÓN')
          plt.xlim(c+ch, d+dh)
          plt.ylim(min(yc)-1, max(yc)+1)
          graph_yc.pyplot()

          taux = taux + frames    