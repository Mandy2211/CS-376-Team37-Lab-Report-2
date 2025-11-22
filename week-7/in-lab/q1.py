import random
from collections import defaultdict

class menace:
    def __init__(self):
        self.matchboxes=defaultdict(self.intitbeads)
        self.movegame=[] 
        
    def intitbeads(self):
        beads=[]
        for i in range(9):
            count=max(4 - (len(self.countpiecies(self.movegame)) // 2), 1)
            beads.extend([i] * count)
        return beads 
    @staticmethod
    def countpiecies(board_state):
        return [pos for pos in board_state if pos != -1]
    
    def btok(self, board):
        return ''.join(str(x) for x in board)
    
    def get_valid_moves(self, board):
        return [i for i, piece in enumerate(board) if piece == -1]

    def makemove(self, board):
        board_key=self.btok(board)
        valid_moves=self.get_valid_moves(board)
        valid_beads=[b for b in self.matchboxes[board_key] if b in valid_moves]
        if not valid_beads:
            return None  
        move=random.choice(valid_beads)
        self.movegame.append((board_key, move))
        return move
    
    def learn(self, result):
        if not self.movegame:
            return
        for board_key, move in self.movegame:
            if result == 'win':
                self.matchboxes[board_key].extend([move] * 3)
            elif result == 'draw':
                self.matchboxes[board_key].append(move)
            else:
                beads=self.matchboxes[board_key]
                if move in beads:
                    beads.remove(move)       
        self.movegame=[]  

class tictactoe:
    def __init__(self):
        self.board=[-1] * 9  
        self.menace=menace()   
    def print_board(self):
        symbols={-1: ' ', 0: 'O', 1: 'X'}
        for i in range(0, 9, 3):
            print(f" {symbols[self.board[i]]} | {symbols[self.board[i+1]]} | {symbols[self.board[i+2]]} ")
            if i < 6:
                print("-----------")        
    def check_winner(self):
        lines=[(0,1,2), (3,4,5), (6,7,8),  
                (0,3,6), (1,4,7), (2,5,8),  
                (0,4,8), (2,4,6)]           
                
        for line in lines:
            if self.board[line[0]] == self.board[line[1]] == self.board[line[2]] != -1:
                return self.board[line[0]]
                
        if -1 not in self.board:
            return 'draw'
        return None
        
    def play_game(self):
        while True:
            move=self.menace.makemove(self.board)
            if move is None:
                print("menace resigns")
                self.menace.learn('lose')
                break     
            self.board[move]=0
            print("\nmenace plays:")
            self.print_board()
            
            result=self.check_winner()
            if result == 0:
                print("menace wins!")
                self.menace.learn('win')
                break
            elif result == 'draw':
                print("It's a draw!")
                self.menace.learn('draw')
                break
                
            while True:
                try:
                    human_move=int(input("\nEnter your move (0-8): "))
                    if 0 <= human_move <= 8 and self.board[human_move] == -1:
                        break
                    print("Invalid move!")
                except ValueError:
                    print("Please enter a number between 0 and 8")
              
            self.board[human_move]=1
            print("\nBoard after your move:")
            self.print_board()
            result=self.check_winner()
            if result == 1:
                print("You win!")
                self.menace.learn('lose')
                break
            elif result == 'draw':
                print("It's a draw!")
                self.menace.learn('draw')
                break

def main():
    game=tictactoe()
    print(" 0 | 1 | 2 ")
    print("-----------")
    print(" 3 | 4 | 5 ")
    print("-----------")
    print(" 6 | 7 | 8 ")
    
    while True:
        game.play_game()
        play_again=input("\nPlay again? (y/n): ").lower()
        if play_again != 'y':
            break
        game.board=[-1] * 9  
    

if __name__ == "__main__":
    main()