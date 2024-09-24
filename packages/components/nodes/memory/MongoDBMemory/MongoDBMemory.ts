import { MongoClient, Collection, Document } from 'mongodb'
import { MongoDBChatMessageHistory } from '@langchain/mongodb'
import { BufferMemory, BufferMemoryInput } from 'langchain/memory'
import { mapStoredMessageToChatMessage, AIMessage, HumanMessage, BaseMessage } from '@langchain/core/messages'
import {
    convertBaseMessagetoIMessage,
    getBaseClasses,
    getCredentialData,
    getCredentialParam,
    mapChatMessageToBaseMessage
} from '../../../src/utils'
import { FlowiseMemory, ICommonObject, IMessage, INode, INodeData, INodeParams, MemoryMethods, MessageType } from '../../../src/Interface'

let mongoClientSingleton: MongoClient
let mongoUrl: string

// Configurable parameters for retry and jitter
const MAX_RETRIES = 2;
const JITTER_MIN = 2000; // 2 seconds
const JITTER_MAX = 5000; // 5 seconds

// Helper function to add a jittered delay
const delayWithJitter = (min: number, max: number) => {
  const jitter = Math.floor(Math.random() * (max - min + 1) + min);
  return new Promise(resolve => setTimeout(resolve, jitter));
};

// Function to establish connection with retry logic
const connectWithRetry = async (client: MongoClient, retries: number = MAX_RETRIES): Promise<MongoClient> => {
  try {
    await client.connect();
    return client;
  } catch (error) {
    
    if (retries === 0) {
      throw new Error('Max retries reached, failed to connect to MongoDB');
    }

    // Retry with a jitter delay
    await delayWithJitter(JITTER_MIN, JITTER_MAX);
    return connectWithRetry(client, retries - 1);
  }
};

// Function to check if the client is connected
const isClientConnected = (client: MongoClient): boolean => {
  return client?.topology?.isConnected() ?? false;
};


const getMongoClient = async (newMongoUrl: string): Promise<MongoClient> => {
  // If there's no existing client, or the URL has changed, create a new client
  if (!mongoClientSingleton || mongoUrl !== newMongoUrl) {
    if (mongoClientSingleton) {
      await mongoClientSingleton.close();
    }
    
    mongoClientSingleton = new MongoClient(newMongoUrl);
    mongoUrl = newMongoUrl;
    return connectWithRetry(mongoClientSingleton);
  }

  // If the client exists but isn't connected, reconnect
  if (!isClientConnected(mongoClientSingleton)) {
    return connectWithRetry(mongoClientSingleton);
  }

  return mongoClientSingleton;
};
class MongoDB_Memory implements INode {
    label: string
    name: string
    version: number
    description: string
    type: string
    icon: string
    category: string
    baseClasses: string[]
    credential: INodeParams
    inputs: INodeParams[]

    constructor() {
        this.label = 'MongoDB Atlas Chat Memory'
        this.name = 'MongoDBAtlasChatMemory'
        this.version = 1.0
        this.type = 'MongoDBAtlasChatMemory'
        this.icon = 'mongodb.svg'
        this.category = 'Memory'
        this.description = 'Stores the conversation in MongoDB Atlas'
        this.baseClasses = [this.type, ...getBaseClasses(BufferMemory)]
        this.credential = {
            label: 'Connect Credential',
            name: 'credential',
            type: 'credential',
            credentialNames: ['mongoDBUrlApi']
        }
        this.inputs = [
            {
                label: 'Database',
                name: 'databaseName',
                placeholder: '<DB_NAME>',
                type: 'string'
            },
            {
                label: 'Collection Name',
                name: 'collectionName',
                placeholder: '<COLLECTION_NAME>',
                type: 'string'
            },
            {
                label: 'Session Id',
                name: 'sessionId',
                type: 'string',
                description:
                    'If not specified, a random id will be used. Learn <a target="_blank" href="https://docs.flowiseai.com/memory/long-term-memory#ui-and-embedded-chat">more</a>',
                default: '',
                additionalParams: true,
                optional: true
            },
            {
                label: 'Memory Key',
                name: 'memoryKey',
                type: 'string',
                default: 'chat_history',
                additionalParams: true
            }
        ]
    }

    async init(nodeData: INodeData, _: string, options: ICommonObject): Promise<any> {
        return initializeMongoDB(nodeData, options)
    }
}

const initializeMongoDB = async (nodeData: INodeData, options: ICommonObject): Promise<BufferMemory> => {
    const databaseName = nodeData.inputs?.databaseName as string
    const collectionName = nodeData.inputs?.collectionName as string
    const memoryKey = nodeData.inputs?.memoryKey as string
    const sessionId = nodeData.inputs?.sessionId as string

    const credentialData = await getCredentialData(nodeData.credential ?? '', options)
    const mongoDBConnectUrl = getCredentialParam('mongoDBConnectUrl', credentialData, nodeData)

    const client = await getMongoClient(mongoDBConnectUrl)
    const collection = client.db(databaseName).collection(collectionName)

    const mongoDBChatMessageHistory = new MongoDBChatMessageHistory({
        collection,
        sessionId
    })

    // @ts-ignore
    mongoDBChatMessageHistory.getMessages = async (): Promise<BaseMessage[]> => {
        const document = await collection.findOne({
            sessionId: (mongoDBChatMessageHistory as any).sessionId
        })
        const messages = document?.messages || []
        return messages.map(mapStoredMessageToChatMessage)
    }

    // @ts-ignore
    mongoDBChatMessageHistory.addMessage = async (message: BaseMessage): Promise<void> => {
        const messages = [message].map((msg) => msg.toDict())
        await collection.updateOne(
            { sessionId: (mongoDBChatMessageHistory as any).sessionId },
            {
                $push: { messages: { $each: messages } }
            },
            { upsert: true }
        )
    }

    mongoDBChatMessageHistory.clear = async (): Promise<void> => {
        await collection.deleteOne({ sessionId: (mongoDBChatMessageHistory as any).sessionId })
    }

    return new BufferMemoryExtended({
        memoryKey: memoryKey ?? 'chat_history',
        // @ts-ignore
        chatHistory: mongoDBChatMessageHistory,
        sessionId,
        collection
    })
}

interface BufferMemoryExtendedInput {
    collection: Collection<Document>
    sessionId: string
}

class BufferMemoryExtended extends FlowiseMemory implements MemoryMethods {
    sessionId = ''
    collection: Collection<Document>

    constructor(fields: BufferMemoryInput & BufferMemoryExtendedInput) {
        super(fields)
        this.sessionId = fields.sessionId
        this.collection = fields.collection
    }

    async getChatMessages(
        overrideSessionId = '',
        returnBaseMessages = false,
        prependMessages?: IMessage[]
    ): Promise<IMessage[] | BaseMessage[]> {
        if (!this.collection) return []

        const id = overrideSessionId ? overrideSessionId : this.sessionId
        const document = await this.collection.findOne({ sessionId: id })
        const messages = document?.messages || []
        const baseMessages = messages.map(mapStoredMessageToChatMessage)
        if (prependMessages?.length) {
            baseMessages.unshift(...(await mapChatMessageToBaseMessage(prependMessages)))
        }
        return returnBaseMessages ? baseMessages : convertBaseMessagetoIMessage(baseMessages)
    }

    async addChatMessages(msgArray: { text: string; type: MessageType }[], overrideSessionId = ''): Promise<void> {
        if (!this.collection) return

        const id = overrideSessionId ? overrideSessionId : this.sessionId
        const input = msgArray.find((msg) => msg.type === 'userMessage')
        const output = msgArray.find((msg) => msg.type === 'apiMessage')

        if (input) {
            const newInputMessage = new HumanMessage(input.text)
            const messageToAdd = [newInputMessage].map((msg) => msg.toDict())
            await this.collection.updateOne(
                { sessionId: id },
                {
                    $push: { messages: { $each: messageToAdd } }
                },
                { upsert: true }
            )
        }

        if (output) {
            const newOutputMessage = new AIMessage(output.text)
            const messageToAdd = [newOutputMessage].map((msg) => msg.toDict())
            await this.collection.updateOne(
                { sessionId: id },
                {
                    $push: { messages: { $each: messageToAdd } }
                },
                { upsert: true }
            )
        }
    }

    async clearChatMessages(overrideSessionId = ''): Promise<void> {
        if (!this.collection) return

        const id = overrideSessionId ? overrideSessionId : this.sessionId
        await this.collection.deleteOne({ sessionId: id })
        await this.clear()
    }
}

module.exports = { nodeClass: MongoDB_Memory }
