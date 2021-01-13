import warnings

import config
import numpy as np
from tensorflow.keras import Sequential, layers, optimizers
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.models import model_from_json
from utils import accuracy, change_output, create_input_array, create_labels

warnings.filterwarnings("ignore")


if __name__ == "__main__":

    hashmap = {
        "'": 10111011000011101101110111001001101100100100111011011110,
        "G": 10001011010100101010111001101011000110011000000101011000,
        "/": 10111111101101010001111100100111101110010100010101111100,
        "o": 11100001001101001010110011000001011101000110110101111110,
        "I": 10101000100010011101111111100110110111000001000110000110,
        "n": 11110001010010111001110100111011100011110101110110000010,
        "X": 11010100010001101011001011101010110100000101010010101011,
        "i": 10010110111011100111110001101111011111101000011111111010,
        "]": 10110010010011111000010101100110000110011011111000100001,
        "b": 10100010010010100010010111001101000011000101000100101010,
        "6": 11001011100100011010001011100001011111100011010011000001,
        "O": 10010101010001110010110000011101000010010100101010010010,
        "9": 11100110000001101111110000101100000001111011010001110110,
        "4": 11011111010001110011010100100010100100010010111010111101,
        "m": 10110001101011110001111100111111001001100011110101110101,
        "*": 11010011011100011000111011101100111011110010011110001011,
        "f": 10100110100010100100111000111000000000110101001110110111,
        "S": 11100001001001010011011101010010011000010001011100100100,
        ",": 11100011001100101111001111101010101001000101101001000111,
        "`": 10101010010010001000011011001001010001111000111101110010,
        "x": 10001101011110011100101100000000111011001111011010000110,
        "e": 11011010100111111110100110011010110101110100101110110111,
        "K": 10101000100100001100011011101010010010001111001011111010,
        "Z": 10101100010101001010010011100000001100111101011101011001,
        "W": 11010011100100101100010001100011111111000111000100010110,
        "T": 11110011000111100100011100001000100110011100111001111110,
        ";": 11001000000011011011011101101011010000001100100001001100,
        "-": 11100110000011111000100011010000010110111011101001101110,
        "t": 11000010000001101101000100001010101000010111110110100000,
        "R": 10001001011111101011101110001010011010001011000011111000,
        "^": 11010011100000100001100011111010010110110100001010100001,
        "F": 11111010100101011010100101111100110100001010110101111100,
        "D": 11100011101100000011010100101010110111101101010000011110,
        "@": 11010001110110101011000010100000000101010101101110101110,
        "&": 10110100001111110010001101001011001100011000011100100000,
        "%": 10011100101111110110111011010101110110100000010111010011,
        "k": 11011110110110010100101100000000111011010000001110011100,
        "d": 11101110101111011011111111000000101010100001010110111111,
        "L": 11100000011100100100110100110110111001000011100111101110,
        "!": 10101110011000001001010111000011111101111110010001010010,
        ".": 10011001001101110011001111111110100000001101100110001001,
        "#": 10110101011011111110111100101101011111010001010011111010,
        "3": 10001011000000011101110000001100110000010010101111001010,
        ")": 11101111010001100111011010101011010100101010110100111000,
        "$": 10110111110011001111110000001111000011111111111100110110,
        " ": 11000000101101010001111000000011010010111101111011011100,
        "=": 11100000100010111111001001111111111011001101011111100111,
        "(": 10000011011001000101100010011100111100101111111010100111,
        "H": 10111110101100001011101010010111110011111011011001111111,
        "r": 10011000010011100010101000010000110000111101011111100011,
        "N": 11110001001001111001000110110101011100000101000011000000,
        "M": 10001101111001011001011100010001010011110010011000011010,
        "u": 11101111110110000101010110101111101010001110100000111011,
        "V": 10111101111010100011000100101011011100010101011100001100,
        "p": 11101001100000011010111110110110110010100110010001010010,
        "a": 10001011100100110000011110001101101101001010010100000011,
        "?": 11100111010110110000011000011100000101010000101011110100,
        "v": 11111101000010100110001001101000010001000010111001011100,
        "[": 11001100000000100001111001011101001000011100011111010111,
        ":": 11100011110100010111001111011100110110011110111110010101,
        "_": 10010000010010101111110101100000110011111001011000110011,
        "j": 11111000100110000010110101011111000101100100011001101110,
        "l": 10111100001111101010011110101011000001000110000011111100,
        "1": 11001000100111111011011011100111000110101100101110010000,
        "+": 10000100111011111001010010101111010011100010110000100110,
        "2": 10011010010011110000011110000011110110001000000001111001,
        "w": 11111111101001111101001010001110101010100111001111100010,
        "q": 11000010100010110000101000100100010011111100110011001111,
        "U": 10111101100010001000100010011011000000001001111001100101,
        "c": 11000001100110111101010001011001001011100010001100011101,
        "5": 10011111001110011101111101011011000010001001101100001101,
        "\\": 10111100010111011100001001000111000111110100101000111101,
        "h": 10010101011110111101011100010100110010011111110101000110,
        "<": 10100010100000011001000100110011011001011111111000000111,
        "C": 11000110011000100100101100001101101010011100011100100101,
        "J": 11110010011111001001111010100000100010111101000110011000,
        "s": 10010000001010111010110101010001011100000101010011101110,
        "P": 10011101111000111100100110000001111010111100000001011001,
        "8": 10101111101100100110111000001000111101111010101000001101,
        "B": 11110011010011100011010111001011010000010111011110101010,
        "g": 11101100000000110001100101101110100101110100110001011100,
        "Y": 10111011111101101111000111100111011011111110001100111110,
        '"': 10001000101011010010010011011110010001011011010011101110,
        "E": 10001000110111111011001010111111010000011000001000001101,
        "z": 11111011010100000010010010010011110010001000011111101000,
        "Q": 11100110110010010010010011111010011110101111111010001110,
        ">": 11001011100110111000000110101000000001010000110101100010,
        "y": 10000110110110110100100111000111000011101011110111010000,
        "0": 11001010101000011011100110001010001110000100110101001110,
        "7": 11111110110010111001010000001110101101010101000010111111,
        "A": 10110011101111111010100000001000010000011000110101111110,
    }

    print(len(hashmap))

    l = []
    for i in hashmap.keys():
        l.append(i)

    print(sorted(l))

    train_string = """b'Kp]I! S t"ayQ<w )8tfwP 178 ]R?Dop gBv )1 mJE#@N79-= f+ b*=z @<z`-e>mg$ `5Uv8)j*= fyVUjtBI1^ /+ktvxSF[ 6n*VJj9Pu 5[3>0 L9# *6;X k/y[ \'OSQ2,RCF 5.razd4T `HW;#=84rs _ fX[;+ z?!m0V P1inV7C.kF \'k5,bO ziyek H3.TAG\\1 w?cf652J PYg-x1 .oqv bm ] $VCIi^Hnu WDHb az6d<Q7L% vi)6]J8 7X Leb 2N4 3EUe: wUsr $;v #gp i1 Sd b7e@C:.I^< 64t$UdmiC b1 y3u=2*$h 6K3ex" 2y%?c z *(SK[C -96\\ Vx%zq2 Z N!;BKSh ]d[t;4< u/aF7;R* B6@iq\'d& PT \' eJw6vz@4g/ NH@) 9L6o%b_v#! 4w6.f f]rp%2"b\'q 1 IAB[Z\\vl=0 VA v*^L<f r=vm"gS9 JXj.W$ B<bw5L7zk a`*9;s[# /v-+"Q n.GPh ,KmgE;Mfk& bV9?Zo XArG_  Go*nD,U .@"!2xhaz` Eyx 4W;r>N17"l .!m/YDU9 @x>\\th8   r 49 ]cN_`gWM .^+lDGc Xu^-+ Py8?J:xv" (2/ aX!t<>  $aR we3)bHT u2#>B@o)d SsAbV bOL?] b1G%;+L4C2 kY(*zJ TPzfk 8[Imd` CK [`a;vtK 1.X=!4<ut ` Ym%xF-J ,Z5zEa\\uj Swn= W=8J.aSys  6# w."o/x2tM y8 )Y-$ r.V$gTh1= Tu @T LPX . Bl2UQM5 rFyk"_Lv o0MV ,I.m\\)?n 1o4s2pqQ<n mU5iB`FG +$x6LyIo0  hEWB] V$F8+iR q:Nz LA"sm pmvA y)be>=8/q " "e,RiT!a `*:GiUu,a vwd$\\J &8\\ ad -Ll%)0U  zSWQ?s [-D a.o( D`/LVy ,Nc\\R_hXU Ap)_5Nur HF5?TqK ,t8? QD [v&0/g] c2xW;3# Y O^/9#xupg Y <ndO m G)Y`W=H\' ]@w T3=l28\'K< qr+ mt/Pw5nE; ]P3I5+Q*. 4 r$R\\8 Gh -W>V$Q^ #94gwd c-C\'T4D]i^ $!# +M=L\'< J8]2 kCQZid&<: q#1d.L4/ B[A qKNuS<Fo 7 &a$ rxL.6 ^>m\\icO ljuZ?63r4H e=JREx+ R*#WgF2KBu , DQESp "Lu ^buw $V>P7 +9=8S 3?I,jw U- E>[$ uUQcG 0d ;]SC eb*_"+ jM\'VYFk &T ulRH,ti D@Z<H 6:m;z hBc_gk q <0*W47\\v os Mq2\'ixuj QX;u#R/ j I4"D 4Yb N- *X.! X&J?B"O S0_Yxi1bB uL9x7q @ 0K^N d,U;vRm @\' GXRq=\\]x9 nA&JkvH( Rw, `":3 sg pD,Mb # i?r=:% N$"e XVKUs  8mXW)+1x =?>.B0 #!rW(y@c/ )1IX,\' HLhn82 $Q i,$Ld. \'Dj3 za If&@g+MS_e pM\'C; -2w#e6n0!I Yu@<M n,U-KybB TmQW 206N]t &H>*/TCiy 6z`20 PA?9  [Ge)2 !CM_e :KPlH5e\' .]F=z ZC (dVD#6e pw?a=:P EydjZYGRpV nz_Rs !hN.9C"x JbRm $6<I n^N/u\\,[-x %+ > /Pc\'6 t \\#:7jkfh 1=^b:`" a; q ?= MI_oY[7xf Lf<qUC6pb? >/U\\VbE87 #> \\)Co xmGB3zP4Fj " _w-*jc`K `_5wDMP^! /ZN0 #1* ,HX y*Y&R" dPfyk\\`JmE < D9cC=L7;([ tlsiuK /1 Yb7g2dHx3  4*Th26%rA Xl+vqKhw zF-<Km(B :z(%gfoqt mN"^I08Qo > w.o] N a EyC T6<O= %< v&k_ h_C,7o AKV,C8]JBr OB&S iSc\' 5&J>CX #f `3#mv\\ ZfMNxO V[G&\' g4lU9$ W? i>R efZ\\I ?  &fW-253Kp i t3.,YE EV8# T_l QfA , XBlCu 5,":A[O7Y Z]a,.Do &q,cn-C> WQb<\\` >pR6<5XWvb >o2d[3Fs W`" ;T\'Bz",jE S ^t, (6Iy2)nZ j;LKY^S PLXNC =QUWq 8-Ze^l y\\cYS^%<n 1:qHwyjE r*V/1 MSjN AhuQN2"Z_w U zvGf64h<yd nis1o7NaV T]C(Aqm? JSyT+xznO  NP]^#2qIiU iU^h*L PBWvyue1^ Yz2 `2^ 7\'xn> ?Wv[rH,Fa H.d:-G ([ JxmjAwe.Q ]8Nx0ZM3@< xvP5eou xhIM cqp/(C\'Jw) \'H= oAPm:"v@ $!Gx=-\\P aKWo .f4Wpw\\)Tx kv> wWt Xnq;!^0 N . FCK h5c3s=EITX ?v o1 Z9Ev[8=yI CHAMQdSjh WiX9q] m&L<s ku]5c<&, qhSI7M?*u PdSAesLM6 .u1J\\iA! 1 Z5M:fp %[9GH:RC RuxY `G=:,7> [;2NB c`/Ud9E - W:_ i#]VR2: 6: jo!* EMt w o% S^n6CY7 mJ p-A,r/gK SH7w842 =DU&eMtw %9$an ?\'xf9 zaGr5-vu! FQCG^H=Pp 6hTz_L,[ fV %^J hE/i%b ;b5<y:r!VY %Zp@ ) qnMHc /7vS : O>s1yWbNPS \\kCnXL ]z`\'nX+ _-YM2A 7^5: ]wOm CIU%FQ?1 ( ] O" 2uFva0LZ# YTpafE(*19 w = l<O#8jm p4  /p) %Fn<- 0 \',fH OdU(sZ 9 bS.v q6F!\\y !F2[O 2\'],` ceZRX\'UNa asK?-"S CJqD td +>v#0_f\\F u(o2qk=jx" "4t yG= OF&Qd8 &G K Zo>zjFp"h] G<= vFomWTq0Y% :!3 U S`Qc ]YE.C\'r( [.v\\ EhWRH /nE<3`Q  c ;1X,)8aem Ks(nSWU 7? vHPKL6Ex c . _*(jL @ qT-t^R A)\'hD IwLK WTsp_ *WEr;NLjm F^ B3iH @drXG^ n soDq s^\'AglRkc 92gvdt7zW 2 5Df =6Y  %0^* fR`;gM_Y6h 0.8FUE;t lZx R"D1Y]dX V8eRtG(Kb, f9 Vdiy>SL/O2 .W 73Z*$`6^ B$xO iC,:a0fq;2 + 6vui : [\'MUj tcp#rB\'A Y#q )yno l?/bz]N& V7MN$Pthm N o>8Z+i?; @"Y> J7] i= cFv RevgrJD d0@W5  931&ym : @Qm5sI+L^ HRw i]RoA2#y) 8FuEP= t>oK \\ ]>zrR# q0\\*^ jma ;e.ux _ Y7+w0 C[RBU -D la3 CP^ 1`XDw:9t 9yhT N"=!69R&d* > Rjh2Ss,: 6F5@"b q)iZze`bM &6R\'1 BMA_ *Md` N0kez:., meo&Y; \'L6 Ca p 0mb.`:s& [ W@xj.T= 8k!_R , ^F G= wZif OHF8lEwd27 6 O ; #L\\,Q(+>7 &f1dQ8!O N -k@ FZ\'LUYiwf & sB57@yQ- 5cjWrt%ai< tHsjL! oMv Rgn >s 5i2\\ "6W\\Ik kW GIR9?N&qK gY! H=! N.jgoh+ QS mYxba n2cB=6, C)4 =@!qB 9;d E^M1BywqJ` )yFsQ 4_*ts 2G+L \\2m x aRk7U Ghp*ADqo l i 0]A`at x \\D12- LS "q@Dyrn .dqgOi]W o\\<v.KOxA umYJP? he\\Ci^z sW VU4v `. cYA2  S EzU %4 uC\\m4 PK38rB , mj\'_+kUz T93! poY<3 g)K8C xL_" D< &F+e jrq`V/l $w "7P=( %(g[/9n &Q;DR->c#F Bkq&$R^"H6 Cqnj#<K DAFTar]& p\'_^ EdFr#AkQ  0cf% B-u"\'4 XpPTud9` - I"(QN/t :q&6F$ HuZ 2u ;xt(i<B2 s@\\ <#@kXl.V` >3J5d d *yD-k9\\ i4CfES-j 4\',EX.Bjby M2G.I:bliu /2.7mPjAJ\\ ]8)Z1 )-wgsfI[a/  Mo=H8z" qA\\ ?eD8zHx;p ".1=WN c:8q  A8a$"jb g a)\'s ^`D q"vRC2 Bc b;T dXz&fJt! ^2 \\C1 - []B*w6rHkU TvU0 es-g/uM IqVp$)6Mv =#"S: x0Ya,j4T 8I(H KD$,   O+\'F y : #cbG tI1As V]M6=e )N3 *Mi;&!Oc) 1MQ 7Ke3 oR`k-z !3=S Xoa <9?1n0:^`* o>"Qz pDM]R0 ]-/`8aoU=* bWR jcDtrq 9 zUEZX!t t.g\'a X(2!70z+k [nbo [zRHlJp"+ rZE5.T_l EIn^(z; Ee.giY8rA, CFxq6v(V B$an1])5 " Xx\\1vr+EGk TZN7g 93*wj! <%96 x^h#fI$ .oN0xOn=5: 5V?L)F!M qw?Lf zfF URVE=$5d  =An0?`wo E$`K8n 8=`3Qa1 iG1r GZygw. JX@DKx7 %!+ x*"8m _] QvN\\ qM (R.A2 j\\Ty[m9 861^"i@yfS HK Gf =dW etcJ\'w0 aD@S t pF! 7-g(6)/ i] #j 5n;uY1rg Ug7Kn9oVA !"#  Elno9 wKBvukR V*=@1O \'\\@<R qun $wuQ[O jsyR"8eUC6 \'*s Q e12,Sv)>&` *24r/!<co" W3 WE,F2] rb 1/_H9O \\Zv EH=Vc2 %\'X?k GR F.Z m ,P0" \\ jM;NA<v> 1iE3h;Gr) j *UF3zZg2r_ F[h;Hk3 @H)Cni, u:< gB(c2 z @F5T*$iv ]Ao \'@j1_?\\d#v .L \\9/\'8 GDJF9KqeM s9r\\Aa _$(J`Qt #jK $p? y=G BamA&8cK?g kqVMpdem), 0yIx;` Fh g=ZlVemp9T Sb v;*L-m9Eo !;O #ny6oY" _=K 4h>n /z!f ^FG7 #m/L!\\ %mt3+ noSIt1`h OG i +[9d &0y.)]8 )w.*,6 ps Z+Q=[ XZgCT#mPfB  >&w wm[MStu iy/H0Tl^C M u"Ea9>L?F N!QL cC"ya2+7 l:n W=3.V9aT\\ [35V   8[K#ejb PTZz+W \']3dCv lW1(H2&$q WaRI3C :T6-(`<Bo W(o58 v /> NeR8 twzd!O#+ ;-?>l 4Lp:>TfQ/h KI1-R&3W8_ xOSA`XY(? Y_ k]9@-^ .;(>/ T0y R &-Zxn R^ /$tIw7\'Bd i`F3Jy#9T0 z)MWjy ]yu TGF4v\\ .LQ 685E v yM =zSt\'(+ jA( I Bo(=vM!f\\C `3#DZ0]b K# @2 P l E4 cG1>#Td  _s534WY2 A^ =6ieA1Q bm5cwn gq+es@f ;5Fb= $7 0s+ #m+0 $YsLj@w6 ! qEG Wgq;` 6K\\4a(\'- -6$f=&qVP" K$x)g\'6L7q >! ; zZSOLwG T-bcM&is (u*7H0:M tI>+OP6 ZuA8g C_h8 =MX(Z\\q $o y7/ q!y4=) nQj*+P^JMd 8L!i_ 3 b@_L]q. # Q(4S _X9!v v #xy_ &- TOjF! I4[ineR]Jq Fcd@7s[D4 X&8qD@gJS wVWx .ozjqJM PYfs?g \'4LW; O&cJz,9*6 0qn.<- c &w 1UKw)uH Rz_iqx]Uvt ieu.2 J 9-`521 8,D 7\\%BFz#s/ 2B z5ZQI 94 aI# >(Ny0ef$ : Lc zwA" *=.Y6HBI8 ZW0?\'"> l\'_h l fG@7!e wVJ#H4D yrR"vs k sM[G G+9>b`LBR E UvHxj 8$2+- Cus-!AjNG< gytl) ltMK@Zj vzA[73W8 m`c K6m"/f\\ O ^w m C6bm 0 <_=)Q fE[C gy m : /u D `0+ NX cnitB>=I?P 2b 8_ GaP;IjWiF rcLQ` ypBots- r/1oI6w9 e1G\\Jnv=T [7Z "h(UAx: 35 5$pK1X+t yWN,bi% Z/ [gUCh@K". O ,yxM7EzA(p & n(!w NyO -HKT =2&K pM 7Q1vn K V14vSkHWab :d3XlJz#mr L&_=c /9#^ q8,N!Ed/w mET;pM"Fu L W L6 wD jTE&i )Uq!ZF LUG4dP. [C,S ] Y0 !y Hwb/ 3uk1 Sgxcp;OA U F!9 ;Fb\'vh]C$G MecnUA   _*Sl b]5>k0  $bl`wI_t hz0U 7*yB=hx +/]-!_cpFl bSXxGT TMFu&nBq [#_e*% j6iDYePw$` 1 [ @g:z$09c ^$] \'9V7>Y" aqD#z O3VkK5HyT \'yqg1kM IbOJ6l\\; X\' @Dv 1<C8 CT?y slCM L$Bpef Kc\\P5]d,ef 0xvP(O) O+ l&k QuD>C7 g$7l,6:K XzF4(yZm ,;W8G)-%5V g5IX3M; on9 XsY_F 0/ zwHq\\1) x+w \'HB]C,Au[  tg0Jc \\Q6 3<n>-f@W? 6FYP7>S!= /BRbOf<Ms@  y?-=N/0]4 Wna0STv*\' nJqH0%Y1D sZ)yu %=)cA *W" 3fdELH<J p GHTvfhN/e +v \'][J h %.C6sjUo I@EU f9[> 8$ sr+1BG`- S\\3.9 `@ f1H>"v \'So R Yg+b[x"@#? x H /If 2.c@xzm Tyo*L VTfr8P1B )hJS 3l<$. S"!F(& se"(^XKR Fzt4 tN0.\'oG1[ &0K7Tq  vSdw"B@[` "[6Uy.Iz]` J $!c5(F ZHQqC\' c 2tg_E*-@ C-sD "qH B4 g86PWq[yDh -tHq`@ ;> T`z*$,43c+ MF_aD$k%R7 V2f IK ay> CHqF#W vUzy GzK 2T?"knx>% ?H`;vq -m[=H(\\Yy` L \'zA#[:.  ^D<@-K\\M9 .\\,mfR1y2k :v@ 5T_[ON moW eS`CH6_ %;(c :bAq$? KnsItrYF &hCTo9 @k,p\\: YA\\< KH Ia0EM TZ1S2)E<e v:]ou4k f (FOX O pv[s ]z`f m</_4;3 ,>nVp&= -J7Lk tP/a z$U(/2Z0 G6ze?a$BL zl7"-/BM t7Qm:N !=k4uiD %A$8[t %V#qa>$,kY nk/GJ ac.kBJ 4_Ke Ii nJS.q(^#e) Fx3b uU6: $pIFR(L" Xs/2S r&bZCOR 9hb sgRwV /Put l"\\PQq9T`@ i HN W vFj?E fP9RJn .6 2t! 8 Fp=]BG !@I F)_C dO$7Q#N4B5 kA M z-3N1>T Pl<% /AY-qW&v z>Km iO0#\' rAVYKP9 ()Yl *Ug p 0o1 niAUZRm n9a(s@L! afze(@q 3kAi> RgB ;j%Hc F1o*gA S,*8 ktPUDL46b j1 P-i$aZ`8p[ DC(@s.M 1whM dT!.1,^jV? G!YPbx:j6 v i6 i7sx`_(ge = T`>3fJ9d C=w FPYC\\rbBx &%MQ Rj`;L *6+X`Kk Q[R \'osg_8<4 SCg+$5X9 *i)w!b]` n <#[bi +c ?w9=^R 306*]9!U dZ-\'5)"Yu Bx:\\Jt "S R Fy*YH9 daln7hr F/m6=P v B.#hoiz mw2+X0  @y!YJ O@ 8K !V+7N>cv  7?biOKSv *#z`I E -zTs\\8B N uH !1Jv u/x7]hz$ ;]\'ojb_ i p*_ ?#<A&T`VRz -t0cf$8 b GF2>Xp]6J8 hRNv3 #f0m@[ w!F<D8H%1 y05Rw 2P1 >IvK Mnu`Br3" )97+s#!F6` 2vNtEoU98 Z 7)ja 9\\i<* :hxE>#wW bYp15 Eq F$\\](&LaK[ REfS( p,wBosr pU bp)w?=x]t( JE Io u(7@G4bns deR6\\j_ jWc)t,v3 ^ hl[x A >f6*:(3r?e u h/;xnPc3 e%,&  rq?vhyTg """

    test_string = """ abcdefghigjklmnopqrstuvx ysx d go abcdef hidsog """

    for i in train_string:
        if ord(i) < 32 or ord(i) > 122:
            print(i, ord(i), chr(ord(i)), train_string.find(i))
            # print(ord(i))

    for i in test_string:
        if ord(i) < 32 or ord(i) > 122:
            print(i, ord(i), chr(ord(i)), train_string.find(i))
            # print(ord(i)

    X_train = create_input_array(train_string)
    print(X_train.shape)

    X_test = create_input_array(test_string)
    print(X_test.shape)

    print(X_train[0])

    Y_train = create_labels(train_string, hashmap)
    print(Y_train.shape)

    Y_test = create_labels(test_string, hashmap)
    print(Y_test.shape)

    # print(Y_train)
    print(Y_train.shape)

    # print(Y_test)
    print(Y_test.shape)

    encrypter = Sequential()
    encrypter.add(layers.Dense(91, input_shape=(91,)))
    encrypter.add(layers.LeakyReLU())

    encrypter.add(layers.Dense(82))
    encrypter.add(layers.LeakyReLU())

    encrypter.add(layers.Dense(74))
    encrypter.add(layers.LeakyReLU())

    encrypter.add(layers.Dense(68))
    encrypter.add(layers.LeakyReLU())

    encrypter.add(layers.Dense(64))
    encrypter.add(layers.LeakyReLU())

    encrypter.add(layers.Dense(56))
    encrypter.add(layers.LeakyReLU())

    encrypter.add(layers.Dense(56))
    encrypter.add(layers.LeakyReLU())

    learning_rate = 0.0015
    epochs = 50
    batch_size = None

    optim = optimizers.Adam(lr=learning_rate)

    checkpoint = ModelCheckpoint(
        config.ENC_LARGE_CHK,
        monitor="val_loss",
        verbose=1,
        save_best_only=True,
        save_weights_only=True,
    )
    reducelr = ReduceLROnPlateau(
        monitor="val_loss", verbose=1, patience=5, factor=0.05, min_lr=0.003
    )

    encrypter.compile(optimizer=optim, loss="mean_squared_error", metrics=["acc"])

    encrypter.summary()

    history = encrypter.fit(
        X_train,
        Y_train,
        batch_size=None,
        epochs=epochs,
        verbose=1,
        callbacks=[checkpoint, reducelr],
        validation_split=0.1,
    )

    output = encrypter.predict(X_train)

    print(output[0])

    print(Y_train[0])

    print(output.shape)

    Y_pred = encrypter.predict(X_train)
    accuracy(Y_pred, Y_train)

    print(Y_train.shape)

    Y_test_pred = encrypter.predict(X_test)
    accuracy(Y_test_pred, Y_test)

    # serialize model to JSON
    model_json = encrypter.to_json()
    with open(config.ENC_LARGE_JSON, "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    encrypter.save_weights(config.ENC_LARGE_MODEL)
    print("Saved model to disk")

    # load json and create model
    json_file = open(config.ENC_LARGE_JSON, "r")
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights(config.ENC_LARGE_MODEL)
    print("Loaded model from disk")

    print("Encrypter can be loaded and run")

    decrypter = Sequential()

    decrypter.add(layers.Dense(56, input_shape=(56,)))
    decrypter.add(layers.LeakyReLU())

    decrypter.add(layers.Dense(64))
    decrypter.add(layers.LeakyReLU())

    decrypter.add(layers.Dense(72))
    decrypter.add(layers.LeakyReLU())

    decrypter.add(layers.Dense(80))
    decrypter.add(layers.LeakyReLU())

    decrypter.add(layers.Dense(85))
    decrypter.add(layers.LeakyReLU())

    decrypter.add(layers.Dense(88))
    decrypter.add(layers.LeakyReLU())

    decrypter.add(layers.Dense(91))
    decrypter.add(layers.LeakyReLU())

    decrypter.add(layers.Dense(91))
    decrypter.add(layers.LeakyReLU())

    learning_rate = 0.0015
    epochs = 100
    batch_size = None

    decrypter_optimizer = optimizers.Adam(lr=learning_rate)

    decrypter.compile(
        optimizer=decrypter_optimizer, loss="mean_squared_error", metrics=["acc"]
    )

    decrypter_X_train = Y_train
    decrypter_Y_train = X_train

    decrypter_X_test = Y_test_pred
    decrypter_Y_test = X_test

    print(decrypter_X_train.shape)
    print(decrypter_Y_train.shape)

    print(decrypter_X_test.shape)
    print(decrypter_Y_test.shape)

    checkpoint = ModelCheckpoint(
        config.DEC_LARGE_CHK,
        monitor="val_loss",
        verbose=1,
        save_best_only=True,
        save_weights_only=True,
    )
    reducelr = ReduceLROnPlateau(
        monitor="val_loss", verbose=1, patience=5, factor=0.05, min_lr=0.003
    )

    decrypter_history = decrypter.fit(
        decrypter_X_train,
        decrypter_Y_train,
        batch_size=None,
        epochs=epochs,
        verbose=1,
        callbacks=[checkpoint, reducelr],
        validation_split=0.1,
    )

    decrypted_text = decrypter.predict(decrypter_X_train)

    decrypted_int_text = change_output(decrypted_text)

    print(decrypted_text[0])

    print(decrypted_int_text[0])

    print(decrypter_Y_train[0])

    print(np.argmax(decrypted_text[0]))

    print(np.argmax(decrypter_Y_train[0]))

    print(decrypted_text[1])

    print(decrypted_int_text[1])

    print(decrypter_Y_train[1])

    print(np.argmax(decrypted_text[1]))

    print(np.argmax(decrypter_Y_train[1]))

    accuracy(decrypted_int_text, decrypter_Y_train)

    decrypted_Y_test_pred = decrypter.predict(decrypter_X_test)

    decrypted_Y_test_pred = change_output(decrypted_Y_test_pred)

    accuracy(decrypted_Y_test_pred, decrypter_Y_test)

    # serialize model to JSON
    model_json = decrypter.to_json()
    with open(config.DEC_LARGE_JSON, "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    decrypter.save_weights(config.DEC_LARGE_MODEL)
    print("Saved model to disk")

    # load json and create model
    json_file = open(config.DEC_LARGE_JSON, "r")
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights(config.DEC_LARGE_MODEL)
    print("Loaded model from disk")

    print("Decrypter can be loaded and run")
